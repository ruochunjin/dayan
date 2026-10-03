import functools
import sys
from enum import Enum
from typing import Union

from dayan.graph.graph import DataFlowGraph, ThoughtGraph

from .instrument import DECORATED_CODE_OBJECTS, tracer_inst
from .invoke import _query_ctx
from .trace import ProgramTrace, global_trace


class AnalysisOutput(Enum):
    TRACE = "trace"
    DATA_FLOW = "data_flow"
    THOUGHT = "thought"


def analysis(func = None, output_type: AnalysisOutput = AnalysisOutput.TRACE):

    if func is None:
        return lambda f: analysis(f, output_type=output_type)

    # [关键] 注册原始函数的代码对象，标记为"白盒"
    # Tracer 会根据此集合决定是否 Step-Into
    DECORATED_CODE_OBJECTS.add(func.__code__)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 上下文管理：增加深度
        is_root = (_query_ctx._depth == 0)
        _query_ctx.enter()

        # 2. 插桩管理：仅在最外层设置 sys.settrace
        previous_trace = None
        if is_root:
            global_trace.clear()  # 新的执行流，清空旧 Trace
            previous_trace = sys.gettrace()
            sys.settrace(tracer_inst.trace_callback)

        try:
            # 执行原函数
            # 注意：如果是嵌套调用，tracer_inst 已经在运行，
            # 它会检测到 func.__code__ 在 DECORATED_CODE_OBJECTS 中，从而自动进入
            result = func(*args, **kwargs)

            # 3. 返回值处理
            if is_root:
                # [目标 4] 最外层返回 Trace
                trace_obj = _build_output_object(global_trace, output_type)
                return result, trace_obj
            else:
                # [目标 3] 内部调用返回原始值，保持逻辑透明
                return result

        finally:
            # 4. 退出逻辑
            if is_root:
                sys.settrace(previous_trace) # 恢复之前的 Tracer

            _query_ctx.exit() # 减少深度

    return wrapper


def _build_output_object(
    trace: ProgramTrace,
    output_type: AnalysisOutput
) -> Union[ProgramTrace, DataFlowGraph, ThoughtGraph]:
    """
    辅助函数：根据 trace 和配置构建最终的返回对象。
    在此处进行 Lazy Import 以防止循环依赖。
    """
    if output_type == AnalysisOutput.TRACE:
        return trace

    # Lazy Import graph modules
    from .graph import DataFlowGraph, ThoughtGraph

    if output_type == AnalysisOutput.DATA_FLOW:
        return DataFlowGraph.from_trace(trace)

    elif output_type == AnalysisOutput.THOUGHT:
        # ThoughtGraph 需要基于 DFG 构建
        dfg = DataFlowGraph.from_trace(trace)
        return ThoughtGraph.build_from_dfg(dfg)

    raise ValueError(f"Unknown output type: {output_type}")
