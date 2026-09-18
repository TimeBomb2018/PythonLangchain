from agent.chat_model import llm

resp = llm.invoke("请用三句话介绍机器学习")
print(type(resp))
print(resp)
