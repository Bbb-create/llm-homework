import whisper

# 加载模型，选base模型，CPU就能跑
model = whisper.load_model("base")

# 识别任务二导出的音频文件
result = model.transcribe("voice_demo.mp3", language="zh")

# 打印识别结果
print("=== 语音识别结果 ===")
print(result["text"])

# 保存结果到文件
with open("asr_result.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
