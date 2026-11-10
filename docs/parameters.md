# 参数文件汇总

全部已发布参数位于 [`quick_start/weights/`](../quick_start/weights/)，运行方式见 [Quick start](../quick_start/README.md)。两个 JSON 索引中的路径均相对于仓库根目录。

## 每折文件

| 文件 | 内容 |
|---|---|
| `eeg_encoder.pt` | EEG 编码器参数，含响应分支、上下文投影、输入中心／尺度及单模态分类器 |
| `eog_encoder.pt` | 眼电／眼动编码器参数，含卷积、BatchNorm、特征读出及单模态分类器 |
| `mapper.pt` | 融合侧卷积池化、角度映射、angle_scale、angle_bias 与 calibration_strength |
| `qjm_head.pt` | 量子门、测量读出、融合权重及分类 MLP |
| `normalizers.pt` | EEG、眼动时序／全局特征及原始眼动输入的归一化统计量和截断值 |
| `config.json` | 输入尺寸、通道／类别顺序、结构配置、精度和校准强度 |

`.pt` 文件仅含张量或张量字典，以 `torch.load(..., weights_only=True)` 读取。各折校准强度保持原值；共享 EEGPT 预训练权重单独保存于 `quick_start/weights/shared/eegpt/`，未计入下表各折 EEG 编码器参数量。

## 各组参数量

全部采用实验基准种子2026；共44套完整模型。

| 模型组 | 折数 | EEG编码器参数量／折 | EOG编码器参数量／折 | 映射器参数量／折 | QJM头参数量／折 |
|---|---:|---:|---:|---:|---:|
| 自有数据集：图片训练 | 12 | 232,596 | 64,807 | 7,273 | 7,252 |
| 自有数据集：视频训练 | 12 | 232,596 | 64,807 | 7,273 | 7,252 |
| SEED-VII | 20 | 233,407 | 64,807 | 7,273 | 7,252 |

参数量按模型Parameter计数，归一化统计量和BatchNorm缓冲区另行保存。EEGPT公共预训练骨干单独保存一次，不计入各折上述EEG编码器参数量。

## 各折目录与校准强度

| 模型组 | 折 | 参数目录 | 推理校准强度 |
|---|---|---|---:|
| own_dataset/image_train | S01 | [S01](../quick_start/weights/own_dataset/image_train/seed2026/S01/) | 0.7 |
| own_dataset/image_train | S02 | [S02](../quick_start/weights/own_dataset/image_train/seed2026/S02/) | 1.1 |
| own_dataset/image_train | S03 | [S03](../quick_start/weights/own_dataset/image_train/seed2026/S03/) | 1 |
| own_dataset/image_train | S04 | [S04](../quick_start/weights/own_dataset/image_train/seed2026/S04/) | 1 |
| own_dataset/image_train | S05 | [S05](../quick_start/weights/own_dataset/image_train/seed2026/S05/) | 0.7 |
| own_dataset/image_train | S06 | [S06](../quick_start/weights/own_dataset/image_train/seed2026/S06/) | 0.6 |
| own_dataset/image_train | S07 | [S07](../quick_start/weights/own_dataset/image_train/seed2026/S07/) | 1 |
| own_dataset/image_train | S08 | [S08](../quick_start/weights/own_dataset/image_train/seed2026/S08/) | 1 |
| own_dataset/image_train | S09 | [S09](../quick_start/weights/own_dataset/image_train/seed2026/S09/) | 1.1 |
| own_dataset/image_train | S10 | [S10](../quick_start/weights/own_dataset/image_train/seed2026/S10/) | 0.5 |
| own_dataset/image_train | S11 | [S11](../quick_start/weights/own_dataset/image_train/seed2026/S11/) | 1.1 |
| own_dataset/image_train | S12 | [S12](../quick_start/weights/own_dataset/image_train/seed2026/S12/) | 0.8 |
| own_dataset/video_train | S01 | [S01](../quick_start/weights/own_dataset/video_train/seed2026/S01/) | 0.8 |
| own_dataset/video_train | S02 | [S02](../quick_start/weights/own_dataset/video_train/seed2026/S02/) | 0.4 |
| own_dataset/video_train | S03 | [S03](../quick_start/weights/own_dataset/video_train/seed2026/S03/) | 0.9 |
| own_dataset/video_train | S04 | [S04](../quick_start/weights/own_dataset/video_train/seed2026/S04/) | 0.9 |
| own_dataset/video_train | S05 | [S05](../quick_start/weights/own_dataset/video_train/seed2026/S05/) | 0.4 |
| own_dataset/video_train | S06 | [S06](../quick_start/weights/own_dataset/video_train/seed2026/S06/) | 0.8 |
| own_dataset/video_train | S07 | [S07](../quick_start/weights/own_dataset/video_train/seed2026/S07/) | 1.1 |
| own_dataset/video_train | S08 | [S08](../quick_start/weights/own_dataset/video_train/seed2026/S08/) | 0.9 |
| own_dataset/video_train | S09 | [S09](../quick_start/weights/own_dataset/video_train/seed2026/S09/) | 0.9 |
| own_dataset/video_train | S10 | [S10](../quick_start/weights/own_dataset/video_train/seed2026/S10/) | 0.7 |
| own_dataset/video_train | S11 | [S11](../quick_start/weights/own_dataset/video_train/seed2026/S11/) | 1 |
| own_dataset/video_train | S12 | [S12](../quick_start/weights/own_dataset/video_train/seed2026/S12/) | 0.8 |
| SEED-VII | S01 | [S01](../quick_start/weights/SEED-VII/seed2026/S01/) | 0.9 |
| SEED-VII | S02 | [S02](../quick_start/weights/SEED-VII/seed2026/S02/) | 0.9 |
| SEED-VII | S03 | [S03](../quick_start/weights/SEED-VII/seed2026/S03/) | 0.9 |
| SEED-VII | S04 | [S04](../quick_start/weights/SEED-VII/seed2026/S04/) | 0.9 |
| SEED-VII | S05 | [S05](../quick_start/weights/SEED-VII/seed2026/S05/) | 0.9 |
| SEED-VII | S06 | [S06](../quick_start/weights/SEED-VII/seed2026/S06/) | 0.9 |
| SEED-VII | S07 | [S07](../quick_start/weights/SEED-VII/seed2026/S07/) | 0.9 |
| SEED-VII | S08 | [S08](../quick_start/weights/SEED-VII/seed2026/S08/) | 0.9 |
| SEED-VII | S09 | [S09](../quick_start/weights/SEED-VII/seed2026/S09/) | 0.9 |
| SEED-VII | S10 | [S10](../quick_start/weights/SEED-VII/seed2026/S10/) | 0.9 |
| SEED-VII | S11 | [S11](../quick_start/weights/SEED-VII/seed2026/S11/) | 0.9 |
| SEED-VII | S12 | [S12](../quick_start/weights/SEED-VII/seed2026/S12/) | 0.9 |
| SEED-VII | S13 | [S13](../quick_start/weights/SEED-VII/seed2026/S13/) | 0.9 |
| SEED-VII | S14 | [S14](../quick_start/weights/SEED-VII/seed2026/S14/) | 0.9 |
| SEED-VII | S15 | [S15](../quick_start/weights/SEED-VII/seed2026/S15/) | 0.9 |
| SEED-VII | S16 | [S16](../quick_start/weights/SEED-VII/seed2026/S16/) | 0.9 |
| SEED-VII | S17 | [S17](../quick_start/weights/SEED-VII/seed2026/S17/) | 0.9 |
| SEED-VII | S18 | [S18](../quick_start/weights/SEED-VII/seed2026/S18/) | 0.9 |
| SEED-VII | S19 | [S19](../quick_start/weights/SEED-VII/seed2026/S19/) | 0.9 |
| SEED-VII | S20 | [S20](../quick_start/weights/SEED-VII/seed2026/S20/) | 0.9 |

各目录内固定为 `eeg_encoder.pt`、`eog_encoder.pt`、`mapper.pt`、`qjm_head.pt`、`normalizers.pt` 和 `config.json`。
完整文件SHA256见 [`parameter_manifest.json`](../quick_start/parameter_manifest.json)；逐个张量的名称和形状见 [`parameter_tensor_index.json`](../quick_start/parameter_tensor_index.json)。
