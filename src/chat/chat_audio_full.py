import base64

import numpy as np
import soundfile as sf

from agent.chat_model import qwen_client

try:
    completion = qwen_client.chat.completions.create(
        model="qwen2.5-omni-7b",
        messages=[{"role": "user", "content": "你是谁"}],
        modalities=["text", "audio"],  # 指定输出文本和音频
        audio={"voice": "Ethan", "format": "wav"},  # "Ethan"、 "Chelsie"
        # modalities= ["text"],
        stream=True,  # 必须设置为 True
        stream_options={"include_usage": True},
    )

    # 3. 处理流式响应并解码音频
    print("模型回复：")
    audio_base64_string = ""
    for chunk in completion:
        # 处理文本部分
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

        # 收集音频部分
        if chunk.choices and hasattr(chunk.choices[0].delta, "audio") and chunk.choices[0].delta.audio:
            audio_base64_string += chunk.choices[0].delta.audio.get("data", "")

    # 4. 保存音频文件
    if audio_base64_string:
        wav_bytes = base64.b64decode(audio_base64_string)
        audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
        sf.write("/Users/xiejunnan/测试/langchain_test/audio_assistant.wav", audio_np, samplerate=24000)
        print("\n音频文件已保存至：audio_assistant.wav")

except Exception as e:
    print(f"请求失败: {e}")