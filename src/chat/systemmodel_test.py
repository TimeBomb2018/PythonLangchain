from agent.chat_model import sys_client

for chunk in sys_client.stream("用三句话简单介绍一下：机器学习的基本概念"):
    # print(type(chunk))
    print(chunk)
    # reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    # print(reasoning_steps if reasoning_steps else chunk.text)
