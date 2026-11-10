# QJM：EEG 与眼电／眼动的量子融合情绪识别

QJM 将 EEG 多尺度响应、EEGPT 上下文和眼电／眼动特征映射到六量子比特电路，通过模态内与跨模态交互完成七类情绪识别。电路采用 PyTorch 复数状态向量实现，可在 CPU 或 CUDA 上运行。

本仓库提供模型源码、推理接口和实验种子为 **2026** 的已发布参数。全部权重与运行示例集中在 [`quick_start/`](quick_start/README.md)；不包含原始数据集、完整训练流程和数据清洗脚本。

## 快速开始

在仓库根目录执行（Python 3.10 或以上）：

```bash
python -m pip install -e .
python quick_start/run.py
```

默认加载自有数据集图片训练的 S01 折。切换至 SEED-VII：

```bash
python quick_start/run.py --fold quick_start/weights/SEED-VII/seed2026/S01
```

示例使用随机信号和随机上下文，只检查权重加载与输出尺寸，**不计算准确率**。真实信号接入与 CUDA 使用见 [Quick start](quick_start/README.md)。

## 目录结构

```text
.
├── src/qjm/                   # 模型源码与 load_fold 推理接口
├── quick_start/
│   ├── README.md              # 运行方法与输入约定
│   ├── run.py                 # 随机输入的加载／尺寸示例
│   ├── weights/
│   │   ├── own_dataset/       # 图片训练、视频训练；各 12 折
│   │   ├── SEED-VII/          # SEED-VII；20 折
│   │   └── shared/eegpt/      # 各折共享的 EEGPT 预训练权重
│   ├── parameter_manifest.json
│   └── parameter_tensor_index.json
├── docs/
│   ├── model.md               # 模型结构与推理流程
│   └── parameters.md          # 参数文件、折目录和校准强度
├── scripts/verify_release.py  # 文件完整性与参数路径校验
├── pyproject.toml             # Python 包安装配置
├── requirements.txt          # 运行依赖
├── THIRD_PARTY_NOTICES.md     # 第三方来源与许可说明
└── SHA256SUMS.txt             # 发布文件的 SHA256 清单
```

## 参数概览

| 模型组 | 折数 | EEG 输入 | 眼电／眼动输入 |
|---|---:|---|---|
| 自有数据集：图片训练 | 12 | 59 × 625，125 Hz，5 秒 | 6 × 625 |
| 自有数据集：视频训练 | 12 | 59 × 625，125 Hz，5 秒 | 6 × 625 |
| SEED-VII | 20 | 62 × 500，125 Hz，4 秒 | 6 × 500 |

共 44 套完整折模型。自有数据集的四个图片／视频训练测试组合复用两个训练方向的模型。每折的通道顺序、类别顺序、精度和校准强度均以该折 `config.json` 为准。

## Python 接口

```python
from qjm import load_fold

model = load_fold(
    "quick_start/weights/own_dataset/image_train/seed2026/S01",
    device="cpu",
)
# eeg、eog 为预处理后、按配置顺序排列且彼此对齐的信号。
# context 为同一批 EEG 对应的 EEGPT 上下文。
logits = model(eeg, eog, context)
```

完整输入约定见 [Quick start](quick_start/README.md)，实现说明见 [模型结构](docs/model.md)。

## 完整性校验

```bash
python scripts/verify_release.py
```

脚本只使用 Python 标准库，检查发布文件 SHA256、折参数清单、张量索引路径和共享 EEGPT 权重。所有清单中的文件路径均相对于仓库根目录。

## 第三方材料

EEGPT 架构参考与公共预训练权重来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

<details>
<summary>🐟 彩蛋：这里藏着一条大肥鱼</summary>

<p align="center">
  <img src="docs/assets/big-fish.png" alt="大肥鱼：蓝发小鲨鱼角色的可爱插画" width="360">
</p>

模型负责识别情绪，大肥鱼负责提供好心情。

</details>
