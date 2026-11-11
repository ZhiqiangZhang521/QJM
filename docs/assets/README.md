# 首页与分享图片

| 文件 | 用途 |
|---|---|
| `deepseek.png` | 用户提供的首页插画原图 |
| `qjm-banner.png` | README 横幅，1600 × 560，完整保留原图主体 |
| `big-fish.png` | README 底部折叠彩蛋的原图 |
| `social-preview.png` | 大肥鱼分享封面，1280 × 640，小于 1 MB |
| `badges/` | README 的五个本地 SVG 徽章，无需外部徽章服务 |

Python 与 PyTorch 徽章标注当前运行版本下限。Qiskit 与天衍量子真机徽章为平台标识；本仓库导出的推理接口使用 PyTorch，不需要安装 Qiskit 或连接真机。

## 设置 GitHub 分享预览图

仓库中的图片文件不会自动成为 GitHub 分享预览图，需要在网页设置中上传：

1. 登录仓库所有者账号，进入 [QJM 的 Settings → General](https://github.com/ZhiqiangZhang521/QJM/settings)。
2. 找到 **Social preview** 区域。
3. 点击图片区域的 **Edit → Upload an image**（未设置过时可能直接显示上传入口）。
4. 选择本目录中的 `social-preview.png`，确认裁剪／保存。
5. 检查该区域显示大肥鱼封面。外部平台已缓存的分享卡片可能需要等待刷新。

README 横幅通过根目录的 `README.md` 引用，无需额外网页设置。
