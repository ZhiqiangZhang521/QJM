# Quick start

本目录保存推理示例、实验种子 **2026** 的 44 折已发布参数、共享 EEGPT 预训练权重和参数索引。以下命令均在仓库根目录执行。

## 1. 安装

使用 Python 3.10 或以上版本，建议在独立环境中安装：

```bash
python -m pip install -e .
```

运行依赖为 PyTorch、NumPy 和 SciPy，版本下限见 [`requirements.txt`](../requirements.txt)。如需 GPU，请使用与本机 CUDA 环境相匹配的 PyTorch。

## 2. 运行示例

```bash
# 自有数据集：图片训练，S01
python quick_start/run.py

# 自有数据集：视频训练，S01
python quick_start/run.py --fold quick_start/weights/own_dataset/video_train/seed2026/S01

# SEED-VII，S01
python quick_start/run.py --fold quick_start/weights/SEED-VII/seed2026/S01

# CUDA 示例
python quick_start/run.py --fold quick_start/weights/SEED-VII/seed2026/S01 --device cuda
```

预期输出包含 `logits shape: (2, 7)`。预测来自随机信号和随机上下文，用于检查加载与尺寸，不代表模型准确率。默认权重路径相对于 `run.py` 定位；显式传入的 `--fold` 相对路径按当前工作目录解释。

## 3. 已发布参数

```text
weights/
├── own_dataset/
│   ├── image_train/seed2026/S01 ... S12
│   └── video_train/seed2026/S01 ... S12
├── SEED-VII/seed2026/S01 ... S20
└── shared/eegpt/
    ├── braindecode_eegpt.safetensors
    └── reference.json
```

每折包含 `eeg_encoder.pt`、`eog_encoder.pt`、`mapper.pt`、`qjm_head.pt`、`normalizers.pt` 和 `config.json`。共享 EEGPT 权重只保存一份。

- [参数文件与各折校准强度](../docs/parameters.md)
- [文件路径、参数量、文件大小和 SHA256](parameter_manifest.json)
- [逐个张量的名称、形状和 dtype](parameter_tensor_index.json)
- [EEGPT 第三方来源](../THIRD_PARTY_NOTICES.md)

两个 JSON 索引中的文件路径均相对于仓库根目录，而不是本目录。

## 4. 接入真实信号

输入必须已经完成清洗、分段和模态对齐，采样率为 125 Hz；不是连续原始采集文件。通道顺序和类别顺序以所选折的 `config.json` 为准。

| 输入 | 自有数据集 | SEED-VII |
|---|---|---|
| `eeg` | `[B, 59, 625]` | `[B, 62, 500]` |
| `eog` | `[B, 6, 625]` | `[B, 6, 500]` |
| `context` | `[B, 39, 4, 512]` | `[B, 31, 4, 512]` |
| 输出 `logits` | `[B, 7]` | `[B, 7]` |

自有数据集的六路眼电为 `HEOR`、`HEOL`、`VEOU`、`VEOL`、`horizontal_diff` 和 `vertical_diff`；SEED-VII 使用左右眼注视坐标和瞳孔直径。两类数据集的类别索引顺序不同，请勿共用一套硬编码标签。

下面的 `eeg` 和 `eog` 应替换为已准备好的、处于同一设备上的浮点张量；EEG 输入单位为微伏：

```python
from qjm import load_fold

model = load_fold(
    "quick_start/weights/own_dataset/image_train/seed2026/S01",
    device="cpu",
)

# 从同一批 EEG 提取上下文；每次调用都会加载共享骨干。
context = model.extract_context(
    eeg,
    "quick_start/weights/shared/eegpt/braindecode_eegpt.safetensors",
)
logits = model(eeg, eog, context)
predictions = logits.argmax(dim=1)
class_names = model.config["class_names"]
labels = [class_names[index] for index in predictions.tolist()]
```

已有匹配的 EEGPT 上下文时，可直接调用 `model(eeg, eog, context)`。上下文必须对应同一批、同一顺序的 EEG 片段。

上下文提取包含跨电极去均值、通道筛选、125 → 256 Hz 重采样及微伏 → 毫伏转换，不负责原始文件清洗、事件切片或眼电提取。

## 5. 使用已提取的特征

`from_features` 接收当前折编码器输出的、尚未进行融合侧标准化的特征：

```python
logits = model.from_features(eeg_features, eog_tokens, eog_global)
```

| 特征 | 自有数据集 | SEED-VII |
|---|---|---|
| `eeg_features` | `[B, 13824]` | `[B, 13888]` |
| `eog_tokens` | `[B, 8, 48]` | `[B, 8, 48]` |
| `eog_global` | `[B, 96]` | `[B, 96]` |

归一化统计量由 `load_fold` 从当前折的 `normalizers.pt` 加载，不应在测试数据上重新拟合。

## 6. 精度与完整性

SEED-VII 原实验使用 CUDA FP16 编码器推理，并将 EEGPT 上下文存为 FP16。自有数据集上下文由 CUDA FP16 计算后存为 FP32。CPU FP32 输出可能与原实验存在数值差异，具体配置见各折 `config.json`。

从仓库根目录验证发布文件：

```bash
python scripts/verify_release.py
```

模型以推理模式加载，所有参数被冻结。`.pt` 文件仅含张量或张量字典，使用 `torch.load(..., weights_only=True)` 读取。
