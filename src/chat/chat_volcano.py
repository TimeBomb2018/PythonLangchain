import asyncio
import time
from volcenginesdkarkruntime import Ark, AsyncArk
from agent.env_utils import VOLCANO_API_KEY, VOLCANO_BASE_URL


volcano_client = Ark(
    base_url=VOLCANO_BASE_URL,
    api_key=VOLCANO_API_KEY
)

async_volcano_client = AsyncArk(
    base_url=VOLCANO_BASE_URL,
    api_key=VOLCANO_API_KEY
)

# 文本理解
# completion = volcano_client.chat.completions.create(
#     # Replace with Model ID
#     model = "doubao-seed-2-0-lite-260215",
#     messages=[
#         {"role": "user", "content": "请将下面内容进行结构化处理：火山方舟是火山引擎推出的大模型服务平台，提供模型训练、推理、评测、精调等全方位功能与服务，并重点支撑大模型生态。 火山方舟通过稳定可靠的安全互信方案，保障模型提供方的模型安全与模型使用者的信息安全，加速大模型能力渗透到千行百业，助力模型提供方和使用者实现商业新增长。"},
#     ],
#     # thinking={"type": "disabled"}, #  Manually disable deep thinking
# )
# print(completion.choices[0].message.content)
#
# 图片理解
# response = volcano_client.responses.create(
#     model="doubao-seed-2-0-lite-260215",
#     input=[
#         {
#             "role": "user",
#             "content": [
#
#                 {
#                     "type": "input_image",
#                     "image_url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png"
#                 },
#                 {
#                     "type": "input_text",
#                     "text": "Which model series supports image input?"
#                 },
#             ],
#         }
#     ]
# )
#
# print(response)

# 视频理解
# async def main():
#     # upload video file
#     print("Upload video file")
#     file = await async_volcano_client.files.create(
#         # replace with your local video path
#         file=open("/Users/xiejunnan/测试/langchain_test/video/demo.mp4", "rb"),
#         purpose="user_data",
#         preprocess_configs={
#             "video": {
#                 "fps": 0.3,  # define the sampling fps of the video, default is 1.0
#             }
#         }
#     )
#     print(f"File uploaded: {file.id}")
#
#     # Wait for the file to finish processing
#     await async_volcano_client.files.wait_for_processing(file.id)
#     print(f"File processed: {file.id}")
#
#     response = await async_volcano_client.responses.create(
#         model="doubao-seed-2-0-lite-260215",
#         input=[
#             {"role": "user", "content": [
#                 {
#                     "type": "input_video",
#                     "file_id": file.id  # ref video file id
#                 },
#                 {
#                     "type": "input_text",
#                     "text": "请你描述下视频中的人物的一系列动作，以JSON格式输出开始时间（start_time）、结束时间（end_time）、事件（event）、是否危险（danger），请使用HH:mm:ss表示时间戳。"
#
#                 }
#             ]},
#         ]
#     )
#     print(response)

# 文档理解
# async def main():
#     # upload pdf file
#     print("Upload pdf file")
#     file = await async_volcano_client.files.create(
#         # replace with your local pdf path
#         file=open("/Users/xiejunnan/测试/langchain_test/document/data.pdf", "rb"),
#         purpose="user_data"
#     )
#     print(f"File uploaded: {file.id}")
#
#     # Wait for the file to finish processing
#     await async_volcano_client.files.wait_for_processing(file.id)
#     print(f"File processed: {file.id}")
#
#     response = await async_volcano_client.responses.create(
#         model="doubao-seed-2-0-lite-260215",
#         input=[
#             {"role": "user", "content": [
#                 {
#                     "type": "input_file",
#                     "file_id": file.id  # ref pdf file id
#                 },
#                 {
#                     "type": "input_text",
#                     "text": "按段落给出文档中的文字内容，以JSON格式输出，包括段落类型（type）、文字内容（content）信息。"
#                 }
#             ]},
#         ],
#     )
#     print(response)

# 图片生成
# imagesResponse = volcano_client.images.generate(
#     # Replace with Model ID
#     model="doubao-seedream-5-0-260128",
#     prompt="充满活力的特写编辑肖像，模特眼神犀利，头戴雕塑感帽子，色彩拼接丰富，眼部焦点锐利，景深较浅，具有Vogue杂志封面的美学风格，采用中画幅拍摄，工作室灯光效果强烈。",
#     size="2K",
#     output_format="png",
#     response_format="url",
#     watermark=False
# )
#
# print(imagesResponse.data[0].url)

response = volcano_client.multimodal_embeddings.create(
    model="doubao-embedding-vision-251215",
    input=[{"type": "text", "text": "Function Calling 是一种将大模型与外部工具和 API 相连的关键功能"}, {"type": "text", "text": "Hello"}],
    encoding_format="float"
)
# 打印结果
print(f"向量维度: {len(response.data.embedding)}")
print(f"前10维向量: {response.data.embedding[:10]}")


# if __name__ == "__main__":
    # asyncio.run(main())

    # 视频生成
    # print("----- create request -----")
    # resp = volcano_client.content_generation.tasks.create(
    #     model="doubao-seedance-1-5-pro-251215",  # Replace with Model ID
    #     content=[
    #         {
    #             "text": (
    #                 "女孩抱着狐狸，女孩睁开眼，温柔地看向镜头，狐狸友善地抱着，镜头缓缓拉出，女孩的头发被风吹动"
    #             ),
    #             "type": "text"
    #         },
    #         {
    #             "image_url": {
    #                 "url": (
    #                     "https://ark-project.tos-cn-beijing.volces.com/doc_image/i2v_foxrgirl.png"
    #                 )
    #             },
    #             "type": "image_url"
    #         }
    #     ],
    #     generate_audio=True,
    #     ratio="adaptive",
    #     duration=5,
    #     watermark=False,
    # )
    #
    # print(resp)
    # 查看视频
    # resp = volcano_client.content_generation.tasks.get(
    #     task_id="cgt-20260324153024-9skqg",
    # )
    # print(resp)

    # 基于首尾帧生成视频
    # print("----- create request -----")
    # create_result = volcano_client.content_generation.tasks.create(
    #     model="doubao-seedance-1-5-pro-251215",  # Replace with Model ID
    #     content=[
    #         {
    #             # Combination of text prompt and parameters
    #             "type": "text",
    #             "text": "图中女孩对着镜头说\"茄子\"，360度环绕运镜"
    #         },
    #         {
    #             # The URL of the first frame image
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seepro_first_frame.jpeg"
    #             },
    #             "role": "first_frame"
    #         },
    #         {
    #             # The URL of the last frame image
    #             "type": "image_url",
    #             "image_url": {
    #                 "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seepro_last_frame.jpeg"
    #             },
    #             "role": "last_frame"
    #         }
    #     ],
    #     generate_audio=True,
    #     ratio="adaptive",
    #     duration=5,
    #     watermark=False,
    # )
    # print(create_result)
    #
    # # Polling query section
    # print("----- polling task status -----")
    # task_id = create_result.id
    # while True:
    #     get_result = volcano_client.content_generation.tasks.get(task_id=task_id)
    #     status = get_result.status
    #     if status == "succeeded":
    #         print("----- task succeeded -----")
    #         print(get_result)
    #         break
    #     elif status == "failed":
    #         print("----- task failed -----")
    #         print(f"Error: {get_result.error}")
    #         break
    #     else:
    #         print(f"Current status: {status}, Retrying after 10 seconds...")
    #         time.sleep(10)
