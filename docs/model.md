# 模型结构与推理接口

## 推理流程

```text
预处理 EEG ── 多尺度响应 ── 三组响应特征 ──┐
         └── EEGPT ─────── 上下文特征 ────┤
                                         ├─ 卷积池化与角度映射
预处理眼电／眼动 ── 时序网络 ── 时序特征 ──┤          │
                          └─ 全局特征 ──┘          ▼
                                               六量子比特电路
                                                     │
                                      EEG／眼动／跨模态三路读出
                                                     │
                                              加权融合与七分类
```

## 源码导航

| 文件 | 职责 |
|---|---|
| [`inference.py`](../src/qjm/inference.py) | `load_fold`、折参数加载、归一化与推理接口 |
| [`response.py`](../src/qjm/response.py) | 多尺度 EEG 响应、能量平滑与压缩 |
| [`eeg.py`](../src/qjm/eeg.py) | EEG 响应与 EEGPT 上下文投影、拼接 |
| [`eegpt.py`](../src/qjm/eegpt.py) | EEGPT 编码器、预训练权重读取与校验 |
| [`eog.py`](../src/qjm/eog.py) | 眼电／眼动多尺度时序网络 |
| [`pools.py`](../src/qjm/pools.py) | 多卷积核时序池化与电极注意力池化 |
| [`fusion.py`](../src/qjm/fusion.py) | 六组特征压缩、角度映射与推理融合头 |
| [`circuit.py`](../src/qjm/circuit.py) | 状态演化、纠缠、读出与分类 |
| [`gates.py`](../src/qjm/gates.py) | RX、RY、RZ 旋转门矩阵 |

## 特征与角度映射

EEG 响应分支采用长度 15、31、63 的可学习时间卷积核，每个尺度包含 4 个滤波器。逐电极计算响应能量，再进行平滑、可学习压缩、相对响应计算和 8 段池化。EEGPT 提供 `[B, patches, 4, 512]` 上下文，每个时间 patch 投影为 64 维。

眼电／眼动分支使用长度 5、11、21 的深度卷积，以及膨胀率 1、2、4 的残差块，产生 `[B, 8, 48]` 时序 tokens 和 `[B, 96]` 全局特征。

融合侧将三组 EEG 响应、一路上下文、一路眼动时序和一路眼动全局特征分别映射为三个旋转角，形成 `[B, 6, 3]`。角度缩放、偏置与校准强度从当前折参数加载。

## 量子电路与读出

前四个量子比特承载 EEG 信息，后两个承载眼电／眼动信息。电路包括局部旋转、两轮模态内 RXX 纠缠与两轮跨模态 RXX 纠缠；跨模态边连接四个 EEG 比特与两个眼动比特。

EEG、眼动、跨模态分支分别读取 16、4、64 维概率分布，经投影和归一化后加权融合。当前发布头将跨模态权重下限设为 0.8，最终 MLP 输出七类 logits。电路以 PyTorch 复数状态向量计算，无需量子硬件。

## 推理接口

- `load_fold(folder, device="cpu")`：严格加载一折参数，设置为冻结的推理模式。
- `model(eeg, eog, context)`：从预处理信号和匹配上下文计算 logits。
- `model.extract_context(eeg, pretrained_path)`：提取 EEGPT 上下文；每次调用重新建立并加载骨干，批量使用时建议预先提取并复用结果。
- `model.from_features(eeg_features, eog_tokens, eog_global)`：从尚未做融合侧标准化的编码器特征计算 logits。
- `return_aux=True`：除 logits 外，返回三路概率、融合权重与跨模态辅助 logits。

## 训练与推理的冻结边界

融合训练阶段，上游 EEGPT 和已训练的 EEG／眼电编码器固定，包括 EEG 上下文投影和眼动全局特征读出；融合侧卷积池化、注意力池化、角度映射、量子门参数、读出及分类器共同优化。

本仓库导出推理接口，不包含重训入口。加载后包括映射器和量子头在内的所有参数均被冻结；`FixedMapper` 描述的是发布后的固定参数映射器。

输入约定见 [Quick start](../quick_start/README.md)，参数明细见 [参数文件汇总](parameters.md)。
