import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# 确保 KeyType 是 Hashable 的，以作为字典的键
KeyType = Union[int, str, Tuple[Any, ...], None]

@dataclass(frozen=True)
class VariableNode:
    """代表数据流图中的一个节点（某个变量的特定版本）"""
    name: str
    scope_id: str  # 用于区分不同函数作用域中的同名变量

    key: KeyType = None
    version: int = 0

    obj_type: type = type(None)  # 记录变量的类型信息
    value: Any = None  # 记录变量的值快照

    is_query_output: bool = False  # 是否为 Query 直接生成的
    is_root_param: bool = False    # 是否为最外层分析函数的参数

    def __repr__(self):
        # 格式化输出: a[0]_v1, d['k']_v2, s{val}_v1
        key_str = ""
        if self.key is not None:
            if isinstance(self.key, int):
                key_str = f"[{self.key}]"
            else:
                key_str = f"[{repr(self.key)}]"
        return f"{self.name}{key_str}@{self.version}"

    def __eq__(self, other):
        return (self.name, self.version, self.scope_id, self.key) == (other.name, other.version, other.scope_id, other.key)

    def __hash__(self):
        return hash((self.name, self.scope_id, self.key, self.version))

@dataclass
class TraceEvent:
    """记录一次定值事件及其依赖"""
    def_node: VariableNode
    use_nodes: List[VariableNode]
    stmt_type: str  # 'assignment', 'call_return', 'arg_pass'
    timestamp: float

@dataclass
class PendingAssignment:
    """记录调用者等待返回值的变量"""
    target_var: str
    scope_id: str

@dataclass
class MutableArgTracker: # TODO: 支持 key 追踪
    """记录参数借出关系：Caller的变量 -> Callee的形参"""
    caller_var: str    # 调用者的变量名 (如 'data')
    caller_scope: str  # 调用者的作用域ID
    callee_arg: str    # 被调用者的参数名 (如 'nums')
    # 核心变更: 记录每个 Key 的初始版本号 {key: version}
    # 对于标量，key 为 None
    key_versions: Dict[KeyType, int] = field(default_factory=dict)

@dataclass
class StackFrameContext:
    """每个调用栈帧的上下文信息"""
    scope_id: str
    # 等待返回值的赋值目标 (b = func())
    return_target: Optional[str] = None
    # 可变参数的映射列表 (func(x) -> x 可能被修改)
    arg_trackers: List[MutableArgTracker] = field(default_factory=list)

class ProgramTrace:
    def __init__(self):
        self.events: List[TraceEvent] = []
        # 记录每个作用域下每个变量的当前版本号
        # [MODIFIED] 版本控制升级为三层结构
        # {scope_id: {var_name: {key: current_version}}}
        self._versions: Dict[str, Dict[str, Dict[KeyType, int]]] = {}

        # 存储 VariableNode，代表之前积累步骤中 USE 的变量
        self._query_buffer_deps: Set[VariableNode] = set()

        # 赋值上下文栈：用于处理 b = func() 的返回流
        self._assignment_stack: List[PendingAssignment] = []

        self._frame_stack: List[StackFrameContext] = []

        self._active_iter_deps: List[VariableNode] = []

    def set_active_iter_deps(self, nodes: List[VariableNode]):
        """在 query_iter 调用时，保存捕获到的 Buffer 依赖"""
        self._active_iter_deps = nodes[:] # 浅拷贝列表

    def get_active_iter_deps(self) -> List[VariableNode]:
        """在循环的每一轮中读取依赖"""
        return self._active_iter_deps

    def push_frame(self, scope_id: str):
        self._frame_stack.append(StackFrameContext(scope_id))

    def pop_frame(self) -> Optional[StackFrameContext]:
        if self._frame_stack:
            return self._frame_stack.pop()
        return None

    def current_frame(self) -> Optional[StackFrameContext]:
        if self._frame_stack:
            return self._frame_stack[-1]
        return None

    def push_assignment(self, var_name: str, scope_id: str):
        self._assignment_stack.append(PendingAssignment(var_name, scope_id))

    def pop_assignment(self) -> Optional[PendingAssignment]:
        if self._assignment_stack:
            return self._assignment_stack.pop()
        return None

    def _get_version_map(self, scope_id: str, var_name: str) -> Dict[KeyType, int]:
        """[Helper] 获取特定变量的所有键版本映射"""
        if scope_id not in self._versions:
            self._versions[scope_id] = {}
        if var_name not in self._versions[scope_id]:
            self._versions[scope_id][var_name] = {}
        return self._versions[scope_id][var_name]

    def get_current_node(self, scope_id: str, var_name: str, key: KeyType = None) -> VariableNode:
        """获取变量的当前版本节点（用于 USE）"""
        v_map = self._get_version_map(scope_id, var_name)
        if key not in v_map:
            v_map[key] = 0
        return VariableNode(var_name, scope_id, key, v_map[key])

    def get_current_version(self, scope_id: str, var_name: str, key: KeyType = None) -> int:
        """获取变量当前的最新版本号，若不存在则返回 -1 或 0"""
        v_map = self._get_version_map(scope_id, var_name)
        return v_map.get(key, 0)

    def new_def_node(self, scope_id: str, var_name: str, key: KeyType = None, val_obj: Any = None, is_query_output: bool = False, is_root_param: bool = False) -> VariableNode:
        """[MODIFIED] 为变量的特定 Key 创建新版本"""
        v_map = self._get_version_map(scope_id, var_name)

        if key not in v_map:
            v_map[key] = 0
        else:
            v_map[key] += 1
        stored_val = val_obj
        if val_obj is not None:
            if isinstance(val_obj, (list, dict, set)):
                try:
                    stored_val = copy.deepcopy(val_obj)
                except Exception:
                    stored_val = val_obj  # 回退到原始对象
            else:
                stored_val = copy.copy(val_obj)

        return VariableNode(var_name, scope_id, key, v_map[key], type(val_obj), stored_val, is_query_output, is_root_param)

    def add_event(self, def_node: VariableNode, use_nodes: List[VariableNode], stmt_type: str):
        event = TraceEvent(def_node, use_nodes, stmt_type, time.time())
        self.events.append(event)

    def buffer_query_dependency(self, nodes: List[VariableNode]):
        """
        [NEW] 当 query 处于积累模式（无返回值）时，记录其使用的变量。
        """
        for node in nodes:
            self._query_buffer_deps.add(node)

    def consume_query_dependencies(self) -> List[VariableNode]:
        """
        [NEW] 当 query 处于触发模式（有返回值）时，提取并清空之前积累的依赖。
        """
        deps = list(self._query_buffer_deps)
        self._query_buffer_deps.clear() # 消费后清空，对应 prompt_buffer.clear()
        return deps

    def clear(self):
        """清空整个 Trace（用于新的执行流）"""
        self.events.clear()
        self._versions.clear()
        self._query_buffer_deps.clear()
        self._assignment_stack.clear()
        self._frame_stack.clear()

# 全局 Trace 对象（实际使用中可能需要上下文管理，这里简化为单例）
global_trace = ProgramTrace()
