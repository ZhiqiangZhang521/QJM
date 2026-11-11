# 首页与分享图片

| 文件 | 用途 |
|---|---|
| `deepseek.png` | 用户提供的首页插画原图 |
| `qjm-banner.png` | README 横幅，2400 × 900，白底淡樱花、英文全称与放大的人物插画 |
| `qjm-banner.svg` | 同版可编辑横幅，文字和图标为矢量，人物插画内嵌 |
| `big-fish.png` | README 底部折叠彩蛋的原图 |
| `social-preview.png` | 大肥鱼分享封面，1280 × 640，小于 1 MB |
| `badges/` | 横幅所用的五个黑色 SVG 图标，无需外部徽章服务 |

五个图标已直接排入横幅，README 下方不再重复显示旧的文字徽章。Python、PyTorch 和 Qiskit 图标来自 Simple Icons v13（CC0）；来源与许可见 [图标说明](icon-sources.md)。Tianyan 使用通用原子示意图，不是平台官方 Logo；Models 使用通用立方体图标。

Qiskit 与天衍量子真机图标为平台标识；本仓库导出的推理接口使用 PyTorch，不需要安装 Qiskit 或连接真机。Python 与 PyTorch 的实际版本下限见根目录安装配置。

## 设置 GitHub 分享预览图

仓库中的图片文件不会自动成为 GitHub 分享预览图，需要在网页设置中上传：

1. 登录仓库所有者账号，进入 [QJM 的 Settings → General](https://github.com/ZhiqiangZhang521/QJM/settings)。
2. 找到 **Social preview** 区域。
3. 点击图片区域的 **Edit → Upload an image**（未设置过时可能直接显示上传入口）。
4. 选择本目录中的 `social-preview.png`，确认裁剪／保存。
5. 检查该区域显示大肥鱼封面。外部平台已缓存的分享卡片可能需要等待刷新。

README 横幅通过根目录的 `README.md` 引用，无需额外网页设置。
