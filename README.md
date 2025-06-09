# In-vehicle-visualization-system
车载可视化系统搭建

这是一个基于 Python 和 Flask 构建的实时情绪识别系统，使用 OpenCV 进行视频流处理，使用深度学习模型进行情绪识别，并通过语音合成提供反馈。该系统可以通过摄像头检测面部情绪，并根据情绪生成车载智能调节建议。

## 功能

- 使用摄像头捕捉视频流并检测人脸。
- 实时情绪识别并生成情绪概率分布。
- 基于情绪生成车载调节方案（语音反馈、环境调节、音乐推荐等）。
- 使用百度语音合成技术提供语音反馈。
- 情绪数据存储在 MySQL 数据库中。

## 📁 项目结构

```
├── templates/
    └── index.html     #  前端页面（情绪图表 + 视频流 + 策略展示）  Flask 模板路径（运行时需放入此目录）
├── LICENSE            #  Apache License 2.0
├── README.md          # 本文件
├── app.py             # 后端主程序
├── enet_b0_8_best_vgaf.onnx       # 情绪识别模型（ONNX 格式）
├── requirements.txt               # Python包依赖清单

```

## 🔧 安装与运行

### 1. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# 或 source venv/bin/activate  # Linux/macOS
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 模型放置

请确认 `enet_b0_8_best_vgaf.onnx` 模型路径与代码中一致，例如：

```python
model_path = r"C:\Users\xxx\your_path\enet_b0_8_best_vgaf.onnx"
```

也可以直接修改为当前目录：

```python
model_path = "enet_b0_8_best_vgaf.onnx"
```

### 4. 启动项目

确保摄像头开启，MySQL 数据库启动，然后运行：

```bash
python real_new_app1.3.py
```



## 🌟 功能亮点

### ✅ 实时情绪识别
- 基于 facenet-pytorch + EmotiEffLib
- 多人脸检测
- 输出主情绪和情绪概率分布

### ✅ 前端可视化
- 条形图：当前情绪分布
- 折线图：情绪随时间变化
- 视频画面叠加主情绪与概率值

### ✅ 多情绪策略生成
- 基于 OpenAI 接口（兼容通义千问 / Qwen 模型）
- 包含：
  - 语音安抚话术（3类风格×2）
  - 环境调节建议（空调、灯光、座椅等）
  - 多媒体推荐（音乐 + 冥想等）

### ✅ 百度语音合成
- 将策略转为 MP3 输出（支持中文情绪语调）

### ✅ 数据存储
- 首次识别的情绪数据保存至 MySQL 数据库

## 📌 注意事项

- 需要本地摄像头可用,无法调用摄像头时请检查摄像头索引或切换摄像头端口；
- 需要运行 MySQL 并配置好 emotion_db 数据库；
- 请放置 HTML 文件于 templates/ 文件夹中；
- 默认使用 CPU，GPU 可改为 "cuda"；

## 📃 License

本项目使用 Apache License 2.0 开源协议。
