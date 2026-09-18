from agent.chat_model import zhipuai_client

with open(r"/Users/xiejunnan/测试/test.wav", "rb") as audio_file:
    resp = zhipuai_client.audio.transcriptions.create(
        model="glm-asr",
        file=audio_file,
        stream=False
    )
    print(resp.model_extra['text'])