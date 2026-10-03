from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

# 引入之前的底层定义
from .trace import ProgramTrace, VariableNode
from .utils import get_logger

logger = get_logger(__name__, console_level=logging.WARNING)

# --- 1. 基础图结构 (DataFlowGraph) ---
# 保持之前的 GraphNode/Edge 定义，这部分不需要大改，它是基础
@dataclass
class GraphEdge:
    source: 'GraphNode'
    target: 'GraphNode'
    relation_type: str
    timestamp: float

class GraphNode:
    def __init__(self, var_node: VariableNode):
        self.var_node = var_node
        self.incoming_edges: List[GraphEdge] = []
        self.outgoing_edges: List[GraphEdge] = []

    @property
    def node_id(self) -> str:
        return repr(self.var_node)

    # 判定是否为 ThoughtGraph 的节点候选
    @property
    def is_thought_node(self) -> bool:
        return self.var_node.is_query_output or self.var_node.is_root_param

    @property
    def label(self) -> str:
        if self.var_node.is_root_param: return "ROOT"
        if self.var_node.is_query_output: return "LLM"
        return "OP" # 普通操作节点

class DataFlowGraph:
    def __init__(self):
        self.nodes: Dict[VariableNode, GraphNode] = {}
        # 辅助：为了拓扑排序，我们需要快速访问
        self.all_nodes: List[GraphNode] = []

    def get_or_create_node(self, var_node: VariableNode) -> GraphNode:
        if var_node not in self.nodes:
            node = GraphNode(var_node)
            self.nodes[var_node] = node
            self.all_nodes.append(node)
        return self.nodes[var_node]

    def add_edge(self, src_var: VariableNode, dst_var: VariableNode, type: str, timestamp: float):
        src_node = self.get_or_create_node(src_var)
        dst_node = self.get_or_create_node(dst_var)
        edge = GraphEdge(src_node, dst_node, type, timestamp)
        src_node.outgoing_edges.append(edge)
        dst_node.incoming_edges.append(edge)

    @classmethod
    def from_trace(cls, trace: ProgramTrace) -> 'DataFlowGraph':

        for event in trace.events:
            # 打印所有的Def 的情况
            logger.debug(f"Processing DEF: {event.def_node}, [is_root: {event.def_node.is_root_param}, is_query_output: {event.def_node.is_query_output}]")
            logger.debug(f"  USEs: {[str(u) for u in event.use_nodes]}")

        graph = cls()
        for event in trace.events:
            # 确保节点存在
            graph.get_or_create_node(event.def_node)
            for use_node in event.use_nodes:
                graph.add_edge(use_node, event.def_node, event.stmt_type, event.timestamp)
        return graph

# --- 2. 核心：属性图 ThoughtGraph ---

class ThoughtGraph:
    """
    属性图实现 G = (V, E, L, A)
    利用 DFG 的 DAG 性质，通过拓扑排序一次性构建 C-path。
    """
    def __init__(self):
        # V: 节点 ID 集合
        self.nodes: Set[str] = set()

        # E: 边集合 (source_id, target_id)
        self.edges: Set[Tuple[str, str]] = set()

        # L: Label 映射 {node_id: 'ROOT' | 'LLM'}
        self.labels: Dict[str, str] = {}

        # A: Attributes 映射 {node_id: {key: val}}
        self.attributes: Dict[str, Dict[str, Any]] = {}

    def add_node(self, node_id: str, label: str, attr_dict: Dict[str, Any]):
        self.nodes.add(node_id)
        self.labels[node_id] = label
        self.attributes[node_id] = attr_dict

    def add_edge(self, src_id: str, dst_id: str):
        self.edges.add((src_id, dst_id))

    @classmethod
    def build_from_dfg(cls, dfg: DataFlowGraph) -> 'ThoughtGraph':
        tg = cls()

        # =========================================================
        # 算法：基于 DAG 的前向依赖传播 (Shadowing Propagation)
        # =========================================================

        # 1. 计算入度 (In-Degree) 准备拓扑排序
        in_degree = {node: len(node.incoming_edges) for node in dfg.all_nodes}
        queue = collections.deque([n for n in dfg.all_nodes if in_degree[n] == 0])

        # 2. 状态表：Nearest Ancestors
        # 映射: DFG_Node -> Set[ThoughtNode_ID]
        # 含义: 到达当前 DFG 节点的所有路径上，最近的 ThoughtGraph 节点集合
        reachability: Dict[GraphNode, Set[str]] = collections.defaultdict(set)

        # 3. 开始拓扑遍历 (Kahn's Algorithm)
        while queue:
            curr = queue.popleft()

            # --- 核心逻辑开始 ---

            # A. 收集上游传递下来的"最近祖先"
            # Union( parent's nearest ancestors )
            incoming_ancestors = set()
            for edge in curr.incoming_edges:
                parent = edge.source
                incoming_ancestors.update(reachability[parent])

            # B. 判断当前节点是否是 ThoughtNode (Is it in V_tg?)
            if curr.is_thought_node:
                curr_id = curr.node_id

                # 1. 将自身加入图 (V, L, A)
                tg.add_node(
                    node_id=curr_id,
                    label=curr.label,
                    attr_dict={
                        "value": curr.var_node.value,
                        "type": curr.var_node.obj_type.__name__,
                        "var_name": curr.var_node.name
                    }
                )

                # 2. 构建边 (E) - 满足 C-path 定义
                # 连接所有逻辑上的直接前驱 -> 当前节点
                for ancestor_id in incoming_ancestors:
                    tg.add_edge(ancestor_id, curr_id)

                # 3. 遮蔽 (Shadowing)
                # 因为 curr 本身是一个 ThoughtNode，它截断了更上游的路径。
                # 对于 curr 的下游节点来说，curr 就是最近的祖先。
                reachability[curr] = {curr_id}

            else:
                # C. 透明传递 (Transparent Propagation)
                # 当前节点是普通运算（如 list.append, +, -），它不阻断思维链
                # 它只是将上游的依赖传递给下游
                reachability[curr] = incoming_ancestors

            # --- 核心逻辑结束 ---

            # 继续拓扑排序
            for edge in curr.outgoing_edges:
                child = edge.target
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return tg

    def print_summary(self):
        print(f"\n{'='*10} Thought Graph Summary {'='*10}")
        print(f"Nodes (|V|): {len(self.nodes)}")
        print(f"Edges (|E|): {len(self.edges)}")

        print("\n[Nodes]")
        for nid in self.nodes:
            lbl = self.labels[nid]
            val = self.attributes[nid]['value']
            print(f"  ({lbl}) {nid} : {val}")

        print("\n[Edges (Logical Dependency)]")
        for u, v in self.edges:
            print(f"  {u} --> {v}")
