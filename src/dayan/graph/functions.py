from typing import Any, Dict, List, Optional, Tuple

from .trace import global_trace


def _find_source_index(target_val: Any, source_pool: List[Tuple[int, Any]]) -> Optional[int]:
    """(辅助函数保持不变) 在源池中寻找与 target_val 匹配的元素索引。"""
    for i, (idx, src_val) in enumerate(source_pool):
        if src_val is target_val:
            source_pool.pop(i); return idx
    for i, (idx, src_val) in enumerate(source_pool):
        if src_val == target_val:
            source_pool.pop(i); return idx
    return None

def handle_sorted(scope_id: str, defs_map: Dict[str, Any], uses_map: Dict[str, Any]) -> bool:
    """
    通用 Handler 签名：
    :param scope_id: 当前作用域 ID
    :param defs_map: 当前行定义的所有变量 {name: object}
    :param uses_map: 当前行使用的所有变量 {name: object}
    :return: Boolean, 是否处理成功
    """

    # 1. 寻找输出目标 (sorted 只返回一个列表)
    # 在 defs_map 中找到第一个列表类型的变量作为 Target
    target_name = None
    target_obj = None

    for name, obj in defs_map.items():
        if isinstance(obj, (list, tuple)):
            target_name = name
            target_obj = obj
            break

    if target_name is None:
        return False # 没找到输出列表，无法处理

    assert target_obj is not None

    # 2. 寻找输入源
    # sorted(iterable, key=..., reverse=...)
    # 我们需要在 uses_map 中找到那个 iterable。
    # 启发式：找到一个 list/tuple，且它的元素能在 target_obj 中找到
    source_name = None
    source_obj = None

    for name, obj in uses_map.items():
        if isinstance(obj, (list, tuple)):
            # 简单的验证：如果 target 非空，检查 target[0] 是否在 obj 里
            if len(target_obj) > 0:
                # 为了性能，只检查第一个元素是否存在
                # 注意：这只是启发式，应对 sorted(a, key=...) 这种场景足够了
                # 如果 a 和 key_func 都是对象，这里排除了 key_func
                try:
                    if target_obj[0] in obj or (obj and target_obj[0] == obj[0]):
                        source_name = name
                        source_obj = obj
                        break
                except Exception: continue
            else:
                # 空列表，直接匹配第一个遇到的列表
                source_name = name
                source_obj = obj
                break

    if source_name is None:
        return False # 没找到输入源

    assert source_obj is not None

    # 3. 执行点对点依赖逻辑
    source_pool = list(enumerate(source_obj))

    for out_idx, out_val in enumerate(target_obj):
        in_idx = _find_source_index(out_val, source_pool)

        if in_idx is not None:
            def_node = global_trace.new_def_node(scope_id, target_name, out_idx, val_obj=out_val)
            use_node = global_trace.get_current_node(scope_id, source_name, in_idx)
            global_trace.add_event(def_node, [use_node], 'reorder')
        else:
            # 无法追踪的数据（可能是新生成的），不做处理或回退
            pass

    return True

# 注册表
SPECIAL_FUNCTION_HANDLERS = {
    'sorted': handle_sorted,
    # 未来支持 zip:
    # 'zip': handle_zip
    # (handle_zip 可以处理 defs_map 中的 tuple 解包, uses_map 中的多个 list)
}
