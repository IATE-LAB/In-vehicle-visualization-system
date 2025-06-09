import cv2
import torch
import pymysql
from datetime import datetime
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from flask import Flask, Response, render_template, jsonify
from contextlib import closing
from aip import AipSpeech
import os
from openai import OpenAI

app = Flask(__name__)

# 初始化摄像头
cap = cv2.VideoCapture(0) #多个摄像头可以换索引为1,2,3...
if not cap.isOpened():
    print("Error: cannot open camera. Check index or device.")
    exit()

# 初始化模型
device = "cpu"  # 或者 "cuda" 如果有 GPU 环境
mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=device)
model_name = get_model_list()[0]
model_path = r"your_path\enet_b0_8_best_vgaf" #后缀一定不要有.onnx
fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_path, device=device)

# MySQL数据库配置
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'emotion_db',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

# 百度语音合成配置
APP_ID = "118389588"
API_KEY = "dVzhcO7oUuUL276KIvZ2wnsn"
SECRET_KEY = "7DlHOR7Ehcqa9AnqvjWw6fNaLTmHEhIT"
client_baidu = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# OpenAI配置
client_openai = OpenAI(
    api_key="sk-c996dcd7166745e9a9a8e143555c4f32",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 全局变量存储第一次识别结果
first_emotion_results = None
first_result_recorded = False
detected_probs = {}  # 用于存储情绪数据

# 初始化MySQL数据库表
def init_mysql_db():
    try:
        with closing(pymysql.connect(**MYSQL_CONFIG)) as conn:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS emotion_records (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp DATETIME NOT NULL,
                        emotion_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                ''')
                conn.commit()
    except pymysql.MySQLError as e:
        print(f"MySQL Error: {e}")
        raise

init_mysql_db()

class EmotionProcessor:
    @staticmethod
    def process_emotion_results(emotion_probs):
        """处理原始情绪概率数据，返回主要情绪"""
        # 过滤低概率情绪（小于10%）
        filtered = {k: v for k, v in emotion_probs.items() if v >= 10}
        if not filtered:
            # 如果没有大于10%的情绪，取前两个最高概率的
            sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_emotions[:2])

        # 转换为中文情绪标签
        emotion_mapping = {
            "happy": "开心",
            "sad": "悲伤",
            "angry": "愤怒",
            "fear": "恐惧",
            "surprise": "惊讶",
            "neutral": "平静",
            "disgust": "厌恶"
        }

        return [{
            "emotion_type": emotion_mapping.get(emotion, emotion),
            "intensity": round(prob / 100, 2)  # 转换为0-1范围
        } for emotion, prob in filtered.items()]

class VehicleStrategyGenerator:
    SYSTEM_PROMPT = """作为车载智能情绪调节系统，请优先提供语音交互方案，并根据情绪状态给出环境调节建议：

【情绪状态】
类型：{emotion_type}
强度：{intensity}/1.0

【方案要求】
1. 语音交互方案（重点）：
   - 3种不同风格的安抚话术（每种风格提供2个版本）
   - 每个话术需包含完整的引导流程（30-50字）
   - 风格包括：温和关怀型、专业指导型、轻松幽默型

2. 环境调节方案（从以下选择至少3项）：
   - 智能空调：温度/风量/模式
   - 氛围灯光：色温（2700-6500K）/亮度/动态效果
   - 座椅设置：按摩强度/通风/加热
   - 车窗开合：开合比例/通风模式
   - 驾驶辅助：建议车速/跟车距离

3. 多媒体方案：
   - 推荐2首适配当前情绪的音乐（注明流派和节奏）
   - 音频内容建议（冥想指导/有声书等）

【输出格式】
### 语音交互方案
[风格类型] 
1. "完整话术内容..."（包含呼吸引导、环境变化说明、后续建议）
2. "替代话术内容..."

### 环境调节方案
1. [设备名称] 具体参数设置...

### 多媒体方案
- 音乐推荐：《曲名》- 歌手（风格，BPM）"""

    @staticmethod
    def generate_prompt(emotion_data):
        # 对每种情绪生成单独的提示词
        prompts = []
        for emotion in emotion_data:
            prompts.append(VehicleStrategyGenerator.SYSTEM_PROMPT.format(
                emotion_type=emotion['emotion_type'],
                intensity=emotion['intensity']
            ))
        return "\n\n".join(prompts)

    @staticmethod
    def get_feedback_strategy(prompt: str) -> str:
        try:
            # 使用流式API
            completion = client_openai.chat.completions.create(
                model="qwen-omni-turbo",
                messages=[
                    {"role": "system", "content": "你是车载语音交互设计专家"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True  # 关键修改：启用流式传输
            )

            # 处理流式响应
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content

            return full_response

        except Exception as e:
            print(f"生成策略时出错: {e}")
            return """### 语音交互方案
[系统默认] 
1. "检测到您当前状态需要放松，建议深呼吸三次：慢慢吸气4秒，屏息2秒，缓缓呼气6秒，我已将空调调整为22℃通风模式，是否需要为您播放舒缓音乐？"
2. "您的安全是我们的首要任务，已启动座椅按摩功能（腰部聚焦模式），前方3公里有服务区，建议稍作休息"

### 环境调节方案
1. [空调] 22℃自动模式
2. [座椅] 开启腰部按摩（强度2级）
3. [多媒体] 播放自然白噪音"""

class VehicleOutputFormatter:
    @staticmethod
    def format_strategy(raw_strategy: str) -> str:
        border = "=" * 40
        return f"\n{border}\n🚘 智能座舱调节方案（语音优先）\n{border}\n{raw_strategy}\n{border}"

def generate_vehicle_strategy(emotion_probs):
    """生成车载调节策略"""
    emotions_data = EmotionProcessor.process_emotion_results(emotion_probs)
    prompt = VehicleStrategyGenerator.generate_prompt(emotions_data)
    strategy = VehicleStrategyGenerator.get_feedback_strategy(prompt)

    # 提取语音内容（删除开头的情绪识别结果）
    voice_lines = []
    lines = strategy.split('\n')
    for line in lines:
        if line.strip().startswith('1. "') or line.strip().startswith('2. "'):
            voice_lines.append(line.strip()[4:].strip('"'))

    voice_text = " ".join(voice_lines)  # 直接连接语音内容，不添加情绪识别结果

    return {
        "strategy": strategy,
        "voice_text": voice_text
    }

def store_emotion_data(emotion_probs):
    """存储情绪数据到MySQL数据库并生成语音"""
    global first_emotion_results, first_result_recorded

    if not first_result_recorded:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emotion_str = ", ".join([f"{k}: {v:.2f}%" for k, v in emotion_probs.items()])

        try:
            with closing(pymysql.connect(**MYSQL_CONFIG)) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO emotion_records (timestamp, emotion_data) VALUES (%s, %s)",
                        (timestamp, emotion_str)
                    )
                    conn.commit()

                    first_emotion_results = emotion_probs
                    first_result_recorded = True

                    print("\n首次识别的多情绪结果:")
                    for emotion, prob in emotion_probs.items():
                        print(f"{emotion}: {prob:.2f}%")

                    # 生成调节策略
                    strategy_data = generate_vehicle_strategy(emotion_probs)
                    print("\n生成的调节策略:")
                    print(VehicleOutputFormatter.format_strategy(strategy_data["strategy"]))

                    # 语音合成（保存到指定路径）
                    text_to_speech(strategy_data["voice_text"],
                                   "D:/PycharmProjects/Visual_project/text/voice/MyVoice.mp3")

        except pymysql.MySQLError as e:
            print(f"存储数据时出错: {e}")

def text_to_speech(text, file_path="D:/PycharmProjects/Visual_project/text/voice/MyVoice.mp3"):
    """将文本转换为语音并保存为MP3文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 调用百度语音合成API
    result = client_baidu.synthesis(text, 'zh', 1, {'vol': 5, 'per': 4})  # per=4是情感女声

    # 保存语音文件
    if not isinstance(result, dict):
        with open(file_path, 'wb') as f:
            f.write(result)
        print(f"\n语音文件已保存到: {file_path}")
    else:
        print("语音合成失败:", result)

def gen_frames():
    """生成处理后的视频帧"""
    global detected_probs
    while True:
        success, frame_bgr = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bboxes, probs = mtcnn.detect(frame_rgb)

        if bboxes is not None and probs is not None:
            for idx, box in enumerate(bboxes):
                if probs[idx] < 0.9:
                    continue
                x1, y1, x2, y2 = box.astype(int)

                # 防越界处理
                h, w, _ = frame_rgb.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_rgb = frame_rgb[y1:y2, x1:x2, :]
                if face_rgb.size == 0:
                    continue

                # 情绪预测
                emotions, scores = fer.predict_emotions([face_rgb], logits=True)
                emotion_top = emotions[0]
                scores_tensor = torch.from_numpy(scores)
                probs_tensor = torch.softmax(scores_tensor, dim=1)
                probs_array = probs_tensor[0].numpy()

                # 更新情绪数据
                detected_probs = {
                    fer.idx_to_emotion_class[i]: prob * 100
                    for i, prob in enumerate(probs_array)
                }

                # 存储第一次识别结果
                if not first_result_recorded:
                    store_emotion_data(detected_probs)

                # 可视化
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, emotion_top, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # 显示概率分布
                line_height = 20
                for e_i, prob_val in enumerate(probs_array):
                    emotion_name = fer.idx_to_emotion_class[e_i]
                    text_str = f"{emotion_name}: {prob_val * 100:.1f}%"
                    text_y = y2 + 20 + e_i * line_height
                    cv2.putText(frame_bgr, text_str, (x1, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_probs')
def emotion_probs():
    # 返回情绪数据
    if detected_probs:
        return jsonify(detected_probs)
    return jsonify({"status": "No emotion data available"})

@app.route('/get_first_result')
def get_first_result():
    if first_emotion_results:
        return jsonify({
            "status": "success",
            "emotions": first_emotion_results
        })
    return jsonify({"status": "pending"})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
