# DaYan 大衍

> 《易经》"大衍之数五十，其用四十有九"——以形式化推演程序生成结构化的卦象状态。**大衍**对大模型推理做同样的事：把推理*过程*变成结构化数据（Thought Graph），再加以分析。

[English README](README.md)

大衍是一个**"思维数据"的建模、评估与增强工具包**。传统评测只看最终答案对不对，大衍回答"模型是怎么推理的"，并把发现回流为改进。

```text
推理程序（您的代码调用 LLM）
  → @analysis 插桩（sys.settrace，不改源代码）
  → 程序 trace → 数据流图 → Thought Graph（节点=LLM 输出的"思维"，边=数据依赖）
  → 过程评估：五个超越准确率的指标
  → 模式挖掘：频繁思维模式（成功 vs 失败对比挖掘）
  → 增强：prompt 规则注入 / 可验证的 RL 过程奖励
```

辅助子包 `dayan.augment` 提供**符号验证的数据增广**（Logical DA）：LLM 生成的样本，标签先经 Z3 / Prover9 / 规则 / CSP 求解器校验才进入训练集。

## 快速开始

```python
from dayan import analysis, query, set_backend, AnalysisOutput

set_backend(my_llm_fn)  # 注入任意 LLM：(context, instruction, temperature) -> str

@analysis(output_type=AnalysisOutput.THOUGHT)
def solve(nums):
    step = query("Add the numbers in {} and get [total]", nums)  # LLM 步骤
    return int(step) * 2

result, thought_graph = solve([4, 4])
```

## 五个过程评估指标

推理效率（频繁模式的分支因子 κ / 首现层 λ）· 关键节点判断（节点影响力相关性 NIS）· 推理共性（路径分布 KL 散度）· 多路径可靠性（路径聚合增益）· 逐步一致性（深度加权 SCS）。

## 可信数据增广

```bash
dayan augment --dataset records.json --output augmented.json --solver z3
```

端点配置：`DAYAN_API_KEY` / `DAYAN_BASE_URL` / `DAYAN_MODEL`（兼容 `OPENAI_*`）。

## 学术脉络

- *Thought Mining for Large Language Model Reasoning*（VLDB 2027）——Thought Graph 数据模型、五个过程指标、两种增强策略
- *Logical DA*（ACL 2025 Findings）——多智能体数据增广与符号标签验证

## 引用本工作

如果您在工作中使用了大衍，请引用相关论文（另见 CITATION.cff）：

```bibtex
@inproceedings{wang2027thought,
  title={Thought Mining for Large Language Model Reasoning},
  author={Wang, Xi and Huang, Zixiao and Zhang, Tailai and Tang, Yuhua and Jin, Ruochun},
  booktitle={Proceedings of the VLDB Endowment (VLDB 2027)},
  year={2027},
  note={Corresponding author: Ruochun Jin}
}

@inproceedings{zheng2025logical,
  title={Logical da: Enhancing data augmentation for logical reasoning via a multi-agent system},
  author={Zheng, Haoqi and Wang, Dong and Yang, Silin and Qi, Yunpeng and Jin, Ruochun and Xu, Liyang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  pages={6843--6855},
  year={2025}
}
```

## 许可

Apache-2.0（见 LICENSE）。
