# LLM 仲裁集成设计方案

本文档阐述如何在当前的 CWRU 轴承故障诊断基线中引入“CNN + 规则贝叶斯网络 + LLM 仲裁”架构，并以 HCAA (Hierarchical Cognitive Arbitration Architecture) 的思路完成训练与评估闭环。

## 1. 总体目标

- 在保持现有 `train.py` 训练流程不变的前提下，引入一个新的评估脚本，对保存的 CNN 模型、BN 规则库以及 LLM 进行联合推理。
- 复用 LLM 仲裁脚本中的 PoE + 温度缩放融合思路，衡量单模型与融合模型在 Accuracy、NLL、ECE、AURC 指标上的表现。
- 通过缓存、配置化的 API Key、可选的采样子集等机制控制 LLM 推理成本。

## 2. 模块划分

| 模块 | 主要职责 | 新增/修改文件 |
| --- | --- | --- |
| 数据载入 | 在测试/验证阶段同时返回原始振动段与谱图 | `data_loader.py`（新增 `return_raw=True` 选项） |
| 特征工程 | 复用/移植 `FeatureExtractor`，对原始信号计算时域与倍频特征 | `utils.py`（新增特征提取函数） |
| 贝叶斯诊断 | 规则式 BN，根据特征生成初步概率分布 | `hcaa/bn_engine.py`（新建） |
| LLM 仲裁 | 组装提示、调用 SiliconFlow DeepSeek-V3，带缓存 | `hcaa/llm_analyzer.py`（新建） |
| 融合与评估 | 读取 CNN 预测、BN、LLM 输出，训练 HCAA 参数并计算指标 | `evaluate_hcaa.py`（新建主脚本） |
| 配置管理 | 管理 API Key、缓存目录、LLM 模型等参数 | `config.py`（新增仲裁相关配置） |

说明：为保持代码组织清晰，建议新增 `hcaa/` 包存放 BN、LLM、融合模型等逻辑。

## 3. 数据管线调整

1. **扩展 Dataset**：
   - 给 `CWRUDataset` 新增布尔参数 `return_raw`（默认 `False`），控制 `__getitem__` 是否额外返回原始时域片段。
   - 训练阶段沿用旧行为，评估脚本使用 `return_raw=True` 以便提取特征。

2. **数据划分重用**：
   - 评估脚本读取 `config.py` 中的 `TRAIN_SPLIT`, `VAL_SPLIT`, `TEST_SPLIT` 配置，通过 `create_dataloaders` 或新增辅助函数获取验证/测试集。
   - 保证 HCAA 训练（在验证集上调参）与最终测试评估与 CNN 训练时采用相同的样本划分。

## 4. 模型与特征

1. **特征提取**：
   - 在 `utils.py` 中实现 `extract_time_features(signal)`、`extract_order_features(signal, fs, shaft_speed=30)`，接口与原脚本保持一致，返回字典结构。
   - 新增 `extract_all_features(signal, fs)` 便于 BN/LLM 调用。

2. **贝叶斯网络**：
   - 将原脚本的 `FaultDiagnosticEngine` 迁移到 `hcaa/bn_engine.py`，并根据我们四类任务更新先验与似然矩阵（如 `['normal', 'ball_fault', 'inner_race', 'outer_race']`）。
   - 保留阈值式证据抽取逻辑，可根据真实数据统计调整阈值。

3. **CNN 预测**：
   - `evaluate_hcaa.py` 中载入 `SimpleCNN`，使用保存的最佳权重 (`results/best_model.pth`) 对验证/测试集谱图做前向，得到 softmax 概率。

## 5. LLM 仲裁模块

1. **API Key 管理**：
   - 在 `config.py` 新增 `SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY", "")`，同时提供 `LLM_MODEL_NAME`, `LLM_CACHE_DIR` 等配置。
   - 评估脚本在缺少 Key 时给出友好错误提示。

2. **缓存策略**：
   - 依据特征 JSON 生成 MD5 哈希，缓存到 `results/llm_cache/<hash>.json`，避免重复请求。
   - 提供 CLI 参数（如 `--skip-llm` 或 `--max-samples`）在调试时跳过 LLM 调用或限制样本数。

3. **提示设计**：
   - 采用结构化 JSON 输出，确保概率与类别顺序一致。
   - 在提示中注入：BN 的 Top-1 预测、关键特征概览（rms、kurtosis、1X/2X 相关指标等）。

## 6. HCAA 融合训练与评估

1. **验证集调参**：
   - `evaluate_hcaa.py` 分别收集验证集上的 CNN、BN、LLM 概率向量及真实标签，使用 `HCAAModel.train`（PoE + 温度缩放）拟合 `alpha`, `beta`, `T`。
   - 支持多专家（例如 CNN、BN、LLM）时，可扩展到 `fused = Π_i p_i^{w_i}`，当前先实现两专家（CNN 与 LLM 或 CNN 与 BN+LLM）。

2. **测试集评估**：
   - 计算 Accuracy、NLL、ECE、AURC 指标，输出表格并保存为 CSV（便于写入 `experiments_log.md`）。
   - 将 CNN-only、BN-only、LLM-only、PoE 未校准、PoE 校准、平均融合等方案统一比较。

3. **结果记录**：
   - 在 `experiments_log.md` 添加新章节，记录 HCAA 评估的设定与结果摘要。

## 7. CLI/脚本接口

- 新建 `evaluate_hcaa.py`，支持以下参数：
  - `--config`: 指定配置文件路径。
  - `--device`: 指定 GPU/CPU。
  - `--max-samples`: 限制每类调用 LLM 的样本数。
  - `--skip-llm`: 仅用缓存或回退到均匀分布，便于离线调试。
- 输出信息包括：进度条、缓存命中率、PoE 学到的参数、指标表等。

## 8. 风险与对策

| 风险 | 应对策略 |
| --- | --- |
| LLM API 调用成本或速率限制 | 提供缓存 + 样本抽样参数，必要时支持离线 JSON 结果注入 |
| 类别标签不一致 | 在 `config.py` 明确 `CLASS_NAMES`，所有模块按统一顺序使用 |
| BN 阈值不匹配真实数据 | 在上线前通过少量标注数据调参，或在配置中暴露阈值 |
| 流水线复杂度提升 | 使用 `hcaa/` 包集中管理新增模块，保持主训练脚本不变 |

## 9. 后续扩展

- 引入不确定性感知的 CNN（如 MC Dropout）作为额外专家。
- 支持多 LLM 模型对比，或在 LLM 提示中加入更多先验知识（工况描述等）。
- 结合实际工厂数据，微调 BN 规则与 LLM 提示模板，进一步提高解释性。

