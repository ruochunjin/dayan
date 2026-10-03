import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .functions import SPECIAL_FUNCTION_HANDLERS
from .trace import KeyType, MutableArgTracker, global_trace
from .utils import get_logger

logger = get_logger(__name__, console_level=logging.INFO)

# 注册被 @analysis 装饰的函数代码对象，用于判断是否 step-into
DECORATED_CODE_OBJECTS: Set[Any] = set()

class CallArgumentParser(ast.NodeVisitor):
    """
    专门解析函数调用处的实际参数（Actual Arguments）。
    目标：提取 func(a, b, x=c) 中的 a, b, c
    """
    def __init__(self, func_name: str):
        self.func_name = func_name
        self.args: List[str] = []      # 位置参数名列表
        self.keywords: Dict[str, str] = {} # 关键字参数 {arg_name: var_name}
        self.found = False

    def visit_Call(self, node):
        # 寻找匹配的函数调用
        # 简单处理：匹配函数名
        # TODO: 处理更复杂的情况，如属性调用 obj.func()
        name_match = False
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            name_match = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == self.func_name:
            name_match = True

        if name_match:
            self.found = True
            # 1. 提取位置参数
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    self.args.append(arg.id)
                else:
                    # 如果实参是字面量(10)或表达式(a+b)，暂记为None，表示无变量依赖
                    self.args.append("")

            # 2. 提取关键字参数
            for keyword in node.keywords:
                if keyword.arg is None:
                    # TODO: 处理 **kwargs 情况
                    continue  # 跳过 **kwargs 情况 WARNING

                if isinstance(keyword.value, ast.Name):
                    self.keywords[keyword.arg] = keyword.value.id
                else:
                    self.keywords[keyword.arg] = ""

        # 继续遍历以防嵌套调用 func(func(x))，这里简单处理只取最外层或第一个匹配
        # TODO: 根据需求决定是否继续深入
        if not self.found:
            self.generic_visit(node)

def get_caller_arguments(source_line: str, func_name: str) -> Tuple[List[str], Dict[str, str]]:
    """解析调用行，返回 (位置实参列表, 关键字实参字典)"""
    try:
        tree = ast.parse(textwrap.dedent(source_line))
        parser = CallArgumentParser(func_name)
        parser.visit(tree)
        return parser.args, parser.keywords
    except Exception:
        return [], {}


MUTATION_METHODS = {
    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'sort', 'reverse', # list
    'update', 'setdefault', 'popitem', # dict
    'add', 'discard', 'difference_update', 'intersection_update' # set
}

class LineParser(ast.NodeVisitor):
    """
    基于 AST 解析一行代码，提取 DEF 变量和 USE 变量名。
    暂不考虑复杂的切片或属性访问，仅处理变量名。
    """
    def __init__(self):
        self.defs: Set[str] = set()
        self.uses: Set[str] = set()
        self.has_query_call: bool = False # 标记当前行是否调用了 query
        self.is_for_loop: bool = False
        self.return_var: Optional[str] = None # [NEW] 记录 return 的变量名
        # [NEW] 记录当前行调用的主要函数名 (用于特殊处理 sorted 等)
        self.called_functions: List[str] = []

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.defs.add(node.id)
        elif isinstance(node.ctx, ast.Load):
            self.uses.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # [NEW] 启发式检测：识别 x.append(...) 这种副作用调用
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in MUTATION_METHODS and isinstance(node.func.value, ast.Name):
                self.defs.add(node.func.value.id)

        # [NEW] 检测函数调用是否为 'query'
        # 简单情况：直接调用 query(...)
        # TODO: 处理更复杂情况，如别名调用等
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            # logger.info(f"Detected function call: {func_name}")
            if func_name in ('query', 'query_iter'):
                self.has_query_call = True

            self.called_functions.append(func_name)

        self.generic_visit(node)

    def visit_arg(self, node):
        self.defs.add(node.arg)

    def visit_Return(self, node):
        # 简单处理：仅支持 return x 形式
        # TODO : 支持 return x + y 等复杂表达式
        if node.value and isinstance(node.value, ast.Name):
            self.return_var = node.value.id
        self.generic_visit(node)

    def visit_For(self, node):
        self.is_for_loop = True
        # Target 是 DEF (例如 x)
        self.visit(node.target)
        # Iter 是 USE (例如 nums)
        self.visit(node.iter)
        # 不继续遍历 body，因为我们只关心这一行 Header

def parse_line_analysis(source_line: str) -> Tuple[Set[str], Set[str], bool, Optional[str], bool, List[str]]:
    """返回 (defs, uses, has_query_call, return_var, is_for_loop, called_functions)"""
    clean_line = textwrap.dedent(source_line).strip()
    if clean_line.startswith(("for ", "while ", "if ", "elif ", "with ", "def ", "class ", "try:", "except:", "else:", "finally:")):
        if not clean_line.endswith(":"): clean_line += ":"
        clean_line += " pass"

    try:
        tree = ast.parse(textwrap.dedent(clean_line))
        parser = LineParser()
        parser.visit(tree)
        # 如果是赋值语句 a = query(...)，parser.defs 会包含 a
        # 如果是表达式语句 query(...)，parser.defs 为空
        return parser.defs, parser.uses, parser.has_query_call, parser.return_var, parser.is_for_loop, parser.called_functions
    except SyntaxError:
        return set(), set(), False, None, False, []

def expand_object_keys(obj: Any) -> List[KeyType]:
    """
    [CRITICAL FIX] 总是返回列表。
    对于空容器，返回 [None] 以便建立对"容器整体"的依赖，防止断链。
    """
    keys: List[KeyType] = []
    if isinstance(obj, list):
        keys = list(range(len(obj)))
    elif isinstance(obj, dict):
        keys = list(obj.keys())
    elif isinstance(obj, set):
        keys = list(obj)
    else:
        # 标量
        return [None]

    # 如果容器为空，返回 [None] 代表依赖容器对象本身
    if not keys:
        return [None]
    return keys

def get_value_by_key(obj: Any, key: KeyType) -> Any:
    if key is None: return obj
    try:
        return obj[key]
    except (IndexError, KeyError, TypeError):
        return None

# --- Deferred Context ---
@dataclass
class DeferredLineContext:
    scope_id: str
    defs: Set[str]
    is_query: bool
    return_var: Optional[str]
    is_for_loop: bool
    called_functions: List[str]
    # [PERFECT FIX] Snapshot: 记录执行前 USE 变量的 Keys
    use_snapshots: Dict[str, List[KeyType]] = field(default_factory=dict)
    use_objects_ref: Dict[str, Any] = field(default_factory=dict)

class ExecutionTracer:
    def __init__(self):
        # 临时存储：用于 Call 事件中捕获赋值目标 (b = func())
        self._temp_call_target: Optional[str] = None
        # 延迟处理缓冲区：{scope_id: DeferredLineContext}
        self._deferred_buffer: Dict[str, DeferredLineContext] = {}

    def _process_deferred_assignment(self, frame, scope_id: str):
        """Line N+1 时刻，利用 Line N 的 Snapshot 和 Line N+1 的 Locals 建立依赖"""

        logger.debug(f"Processing deferred assignment for scope {scope_id}.")

        if scope_id not in self._deferred_buffer:
            return

        ctx = self._deferred_buffer.pop(scope_id)

        try:
            active_special_handler = None
            for f_name in ctx.called_functions:
                if f_name in SPECIAL_FUNCTION_HANDLERS:
                    active_special_handler = SPECIAL_FUNCTION_HANDLERS[f_name]
                    break

            # 如果存在特殊处理函数，且有定义变量
            if active_special_handler and ctx.defs:

                # A. 构建 DEFs Map (当前时刻的真实值)
                defs_map = {}
                for def_name in ctx.defs:
                    if def_name in frame.f_locals:
                        defs_map[def_name] = frame.f_locals[def_name]

                # B. 获取 USEs Map (执行前的对象引用)
                uses_map = ctx.use_objects_ref

                # C. 传递全量上下文给 Handler
                if defs_map and uses_map:
                    # 接口变更：传入 Map，让 Handler 自己决定取哪个变量
                    handled = active_special_handler(scope_id, defs_map, uses_map)

                    if handled:
                        return # 成功处理，跳过后续通用逻辑

            # 1. 构建 USE 节点 (基于 Snapshot)
            # 因为变量在 Line N 执行过程中可能被修改(e.g. pop)，所以必须用执行前的 Keys
            current_use_nodes = []
            is_llm_def = False

            # [Case A] Query Iter Loop (for x in query_iter(...))
            if ctx.is_for_loop and ctx.is_query:
                is_llm_def = True
                stmt_type = 'llm_iter'

                # 1. 获取显式参数依赖 (Explicit Args)
                # 这些是通过 AST 解析出来的 query_iter(...) 中的参数
                # 每一轮循环，instrument 都会重新快照这些参数，所以总是最新的
                for use_var, keys in ctx.use_snapshots.items():
                    for key in keys:
                        current_use_nodes.append(global_trace.get_current_node(scope_id, use_var, key))

                # 2. 获取隐式上下文依赖 (Context Buffer)
                # 从 trace 的暂存区读取。这份数据在 query_iter 调用时被锁定，不会随循环清空。
                buffered_deps = global_trace.get_active_iter_deps()
                current_use_nodes.extend(buffered_deps)

                for def_name in ctx.defs:
                    if def_name not in frame.f_locals: continue
                    def_obj = frame.f_locals[def_name]
                    def_keys = expand_object_keys(def_obj)
                    for key in def_keys:
                        val = get_value_by_key(def_obj, key)
                        def_node = global_trace.new_def_node(
                            scope_id, def_name, key, val_obj=val, is_query_output=is_llm_def
                        )
                        global_trace.add_event(def_node, current_use_nodes, stmt_type)

            # [Case B] Single Query (x = query(...))
            elif ctx.is_query:
                stmt_type = 'llm_invoke'
                is_llm_def = True
                # 1. 获取显式参数依赖 (Explicit Args)
                # Query 模式：依赖 Snapshot 中的所有 keys
                line_uses = []
                for use_var, keys in ctx.use_snapshots.items():
                    # 注意：get_current_node 获取的是当前版本，Key 结构来自 Snapshot
                    for key in keys:
                        line_uses.append(global_trace.get_current_node(scope_id, use_var, key))

                if not ctx.defs:
                    # Accumulate Mode
                    global_trace.buffer_query_dependency(line_uses)
                    return # 结束

                # Invoke Mode : 消费 Buffer + 当前参数
                buffered_deps = global_trace.consume_query_dependencies()
                current_use_nodes = line_uses + buffered_deps

                for def_name in ctx.defs:
                    if def_name not in frame.f_locals: continue
                    def_obj = frame.f_locals[def_name]
                    def_keys = expand_object_keys(def_obj)
                    for key in def_keys:
                        val = get_value_by_key(def_obj, key)
                        def_node = global_trace.new_def_node(
                            scope_id, def_name, key, val_obj=val, is_query_output=True
                        )
                        global_trace.add_event(def_node, current_use_nodes, stmt_type)

            # [Case C] 普通 For Loop (for x in nums)
            elif ctx.is_for_loop:
                stmt_type = 'loop_iter'

                # 准备迭代器信息 (Assuming single iterator for simplicity)
                iterator_var = list(ctx.use_snapshots.keys())[0]
                iterator_obj = ctx.use_objects_ref.get(iterator_var)

                for def_name in ctx.defs:
                    if def_name not in frame.f_locals: continue
                    def_obj = frame.f_locals[def_name] # Current Loop Variable Value

                    # 循环变量通常是标量，但如果是 for a,b in ... 则可能是解包
                    # 这里简化处理：假设 def_obj 本身就是我们要找的值

                    # [反向查找逻辑]
                    matched_key = None
                    if iterator_obj is not None:
                        if isinstance(iterator_obj, list):
                            try:
                                # 优先 Identity
                                for i, item in enumerate(iterator_obj):
                                    if item is def_obj:
                                        matched_key = i; break
                                # 其次 Equality
                                if matched_key is None:
                                    matched_key = iterator_obj.index(def_obj)
                            except Exception: pass
                        elif isinstance(iterator_obj, dict):
                            if def_obj in iterator_obj:
                                matched_key = def_obj

                    # 构建依赖
                    current_deps = []
                    if matched_key is not None:
                        current_deps.append(global_trace.get_current_node(scope_id, iterator_var, matched_key))
                    else:
                        # Fallback: 依赖整体
                        current_deps.append(global_trace.get_current_node(scope_id, iterator_var, None))

                    # 创建节点
                    # 循环变量的 keys 通常是 [None] (标量)，除非它是 tuple 解包
                    def_keys = expand_object_keys(def_obj)
                    for key in def_keys:
                        val = get_value_by_key(def_obj, key)
                        def_node = global_trace.new_def_node(
                            scope_id, def_name, key, val_obj=val, is_query_output=False
                        )
                        global_trace.add_event(def_node, current_deps, stmt_type)
            # [Case D] 普通赋值
            else:
                stmt_type = 'assignment'

                for def_name in ctx.defs:
                    if def_name not in frame.f_locals: continue
                    def_obj = frame.f_locals[def_name]
                    def_keys = expand_object_keys(def_obj)

                    for key in def_keys:
                        val = get_value_by_key(def_obj, key)
                        def_node = global_trace.new_def_node(
                            scope_id, def_name, key, val_obj=val, is_query_output=False
                        )

                        current_deps = []

                        # [策略 B] 点对点匹配 (Point-to-Point Matching)
                        # 条件: 只有 1 个 USE 变量 (e.g. slicing, copy)
                        if len(ctx.use_snapshots) == 1:
                            use_name = list(ctx.use_snapshots.keys())[0]
                            use_keys_snapshot = ctx.use_snapshots[use_name]

                            if key in use_keys_snapshot:
                                # Hit! DEF[k] <- USE[k]
                                current_deps.append(global_trace.get_current_node(scope_id, use_name, key))
                            else:
                                # Miss: Fallback to Broadcast
                                current_deps.append(global_trace.get_current_node(scope_id, use_name, None))

                        # [策略 C] 广播 / 表达式计算 (Broadcasting)
                        else:
                            for use_var in ctx.use_snapshots.keys():
                                current_deps.append(global_trace.get_current_node(scope_id, use_var, None))

                        global_trace.add_event(def_node, current_deps, stmt_type)

            # 3. 处理 Return 传递 (如果该行包含 return 语句)
            if ctx.return_var:
                # 检查外部是否有等待赋值的 pending
                pending = global_trace.pop_assignment()
                if pending and ctx.return_var in frame.f_locals:
                    ret_obj = frame.f_locals[ctx.return_var]
                    ret_keys = expand_object_keys(ret_obj)

                    for key in ret_keys:
                        # Def(Caller.Target) <- Use(Callee.ReturnVar)
                        val = get_value_by_key(ret_obj, key)
                        def_node = global_trace.new_def_node(pending.scope_id, pending.target_var, key, val)
                        use_node = global_trace.get_current_node(scope_id, ctx.return_var, key)
                        global_trace.add_event(def_node, [use_node], 'return_pass')

        except Exception as e:
            # print(f"Deferred Error: {e}")
            pass

        logger.debug(f"Finished processing deferred assignment for scope {scope_id}.")

    def trace_callback(self, frame, event, arg):
        code = frame.f_code
        scope_id = str(id(frame))

        # [Rule 1] 任何事件触发，先处理上一行遗留的 Deferred 任务
        # 这确保了 Line N 的赋值在 Line N+1 (或 return/call) 之前被记录
        self._process_deferred_assignment(frame, scope_id)

        # ---------------------------------------------------------
        # 1. CALL 事件
        # ---------------------------------------------------------
        if event == 'call':
            logger.debug(f"CALL event in {code.co_name}, function call is {code}.")
            if code in DECORATED_CODE_OBJECTS:

                is_entry_point = (len(global_trace._frame_stack) == 0)

                global_trace.push_frame(scope_id)
                curr_ctx = global_trace.current_frame()

                assert curr_ctx is not None, "Current frame context should not be None after push_frame."

                # 处理临时赋值目标
                if self._temp_call_target:
                    curr_ctx.return_target = self._temp_call_target
                    parent_scope = str(id(frame.f_back)) if frame.f_back else "unknown"
                    global_trace.push_assignment(self._temp_call_target, parent_scope)
                    self._temp_call_target = None

                # 解析参数传递
                func_name = code.co_name
                arg_info = inspect.getargvalues(frame)
                formal_args = arg_info.args
                caller_frame = frame.f_back

                if caller_frame:
                    try:
                        caller_scope = str(id(caller_frame))
                        lines, start_line = inspect.getsourcelines(caller_frame.f_code)
                        lineno = caller_frame.f_lineno - start_line
                        if 0 <= lineno < len(lines):
                            caller_line = lines[lineno]
                            actual_args, actual_kwargs = get_caller_arguments(caller_line, func_name)

                            for i, formal_name in enumerate(formal_args):
                                if formal_name == 'self': continue

                                actual_name = None
                                if i < len(actual_args): actual_name = actual_args[i]
                                elif formal_name in actual_kwargs: actual_name = actual_kwargs[formal_name]

                                if actual_name and actual_name in caller_frame.f_locals and formal_name in frame.f_locals:
                                    # 注意：Snapshot 是基于 Caller 视角的实参结构
                                    arg_obj = frame.f_locals[formal_name]
                                    keys = expand_object_keys(arg_obj)

                                    tracker = MutableArgTracker(
                                        caller_var=actual_name, caller_scope=caller_scope, callee_arg=formal_name
                                    )

                                    for key in keys:
                                        # Arg Pass Event
                                        use_node = global_trace.get_current_node(caller_scope, actual_name, key)
                                        val = get_value_by_key(arg_obj, key)
                                        def_node = global_trace.new_def_node(scope_id, formal_name, key, val)
                                        global_trace.add_event(def_node, [use_node], 'arg_pass')

                                        # 记录进入时的版本 (用于 Side Effect 检测)
                                        tracker.key_versions[key] = global_trace.get_current_version(scope_id, formal_name, key)

                                    curr_ctx.arg_trackers.append(tracker)
                    except Exception:
                        pass

                if is_entry_point:
                    for formal_name in formal_args:
                        if formal_name == 'self': continue
                        if formal_name in frame.f_locals:
                            arg_obj = frame.f_locals[formal_name]
                            keys = expand_object_keys(arg_obj)
                            for key in keys:
                                val = get_value_by_key(arg_obj, key)
                                # [MODIFIED] 显式标记 is_root_param=True
                                # 注意：如果是递归调用的入口(frame_stack!=0)，这里不应该标记
                                # new_def_node 会增加版本(v0)，并记录为 ROOT
                                def_node = global_trace.new_def_node(
                                    scope_id, formal_name, key, val_obj=val,
                                    is_root_param=True
                                )
                                global_trace.add_event(def_node, [], 'root_param')

                logger.debug(f"END OF CALL event in {code.co_name}.")
                return self.trace_callback
            logger.debug(f"END OF CALL event in {code.co_name}.")
            return None

        # ---------------------------------------------------------
        # 2. RETURN 事件
        # ---------------------------------------------------------
        elif event == 'return':
            # [CRITICAL FIX] 函数即将结束，必须立刻处理当前 Scope 剩余的 Deferred 任务
            # 因为不会再有下一行 Line 事件来触发它了
            logger.debug(f"RETURN event in {code.co_name}, code is {code}.")
            self._process_deferred_assignment(frame, scope_id)

            ctx = global_trace.pop_frame()
            if ctx:
                # A. Side Effect Detection
                for tracker in ctx.arg_trackers:
                    if tracker.callee_arg in frame.f_locals:
                        final_obj = frame.f_locals[tracker.callee_arg]
                        current_keys = expand_object_keys(final_obj)

                        # 检查 Keys 的并集：捕获新增、修改、删除
                        all_keys = set(tracker.key_versions.keys()) | set(current_keys)

                        for key in all_keys:
                            is_changed = False

                            # 1. Key 被删除
                            if key in tracker.key_versions and key not in current_keys:
                                is_changed = True
                            # 2. Key 新增
                            elif key not in tracker.key_versions and key in current_keys:
                                is_changed = True
                            # 3. Key 值改变 (版本增加)
                            elif key in current_keys:
                                final_ver = global_trace.get_current_version(scope_id, tracker.callee_arg, key)
                                if final_ver > tracker.key_versions[key]:
                                    is_changed = True

                            if is_changed:
                                val = get_value_by_key(final_obj, key)
                                def_node = global_trace.new_def_node(tracker.caller_scope, tracker.caller_var, key, val)
                                # Side effect 依赖于函数结束时的状态
                                use_node = global_trace.get_current_node(scope_id, tracker.callee_arg, key)
                                global_trace.add_event(def_node, [use_node], 'side_effect')

                # B. Stack Cleanup
                # 如果 Return 语句未被 Line Parser 捕获（罕见），或隐式 Return None
                # 必须清理 Pending Assignment，防止栈错位
                global_trace.pop_assignment()
            logger.debug(f"END OF RETURN event in {code.co_name}.")
            return self.trace_callback

        # ---------------------------------------------------------
        # 3. LINE 事件
        # ---------------------------------------------------------
        elif event == 'line':
            logger.debug(f"LINE event in {code.co_name}, code is {code}.")
            try:
                lines, start_line = inspect.getsourcelines(code)
                lineno = frame.f_lineno - start_line
                if 0 <= lineno < len(lines):
                    source_line = lines[lineno]
                    defs, uses, is_query, return_var, is_for_loop, called_funcs = parse_line_analysis(source_line)

                    # [PERFECT FIX] Snapshot Logic
                    # 在代码执行前，记录所有 USE 变量的 keys
                    use_snapshots = {}
                    use_objects_ref = {}

                    for use_var in uses:
                        # 查找变量对象
                        obj = frame.f_locals.get(use_var, frame.f_globals.get(use_var))
                        # 记录 Keys 快照
                        if obj is not None:
                            use_snapshots[use_var] = expand_object_keys(obj)
                            use_objects_ref[use_var] = obj
                        else:
                            # 变量可能尚未定义或不可达
                            use_snapshots[use_var] = [None]

                    # 记录 Call 目标预判
                    if defs:
                        self._temp_call_target = list(defs)[0]
                    else:
                        self._temp_call_target = None

                    # 注册 Deferred Context
                    # 只有当存在定义、是query调用或有返回值时才需要处理
                    if defs or is_query or return_var:
                        self._deferred_buffer[scope_id] = DeferredLineContext(
                            scope_id=scope_id,
                            defs=defs,
                            is_query=is_query,
                            return_var=return_var,
                            use_snapshots=use_snapshots, # <-- 存入快照
                            use_objects_ref=use_objects_ref, # <-- 存入对象引用
                            is_for_loop=is_for_loop,
                            called_functions=called_funcs
                        )

            except Exception:
                logger.debug(f"LINE event processing error in {code.co_name}.")
                pass
            logger.debug(f"END OF LINE event in {code.co_name}.")
            return self.trace_callback

        return self.trace_callback

tracer_inst = ExecutionTracer()
