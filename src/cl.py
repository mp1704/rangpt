import json
import chainlit as cl
import requests
import time
from handle_conversation import handle_conversation_turn, add_prompt_to_conversation

@cl.on_chat_start
def start_chat():
    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
    
    cl.user_session.set(
        "conversation_actual",
        [{"role": "system", "content": system_prompt}],
    )
    cl.user_session.set("conversation_history", "")
    
@cl.on_message
async def main(message: cl.Message):
    start = time.time()
    conversation_history = cl.user_session.get("conversation_history")
    conversation_actual = cl.user_session.get("conversation_actual") # list
    
    if message.content.lower() == "quit": pass
    
    conversation_history += f"User: {message.content}\n"
    conversation_actual.append({"role": "user", "content": message.content })
    response = handle_conversation_turn(conversation_history, conversation_actual)
    conversation_history += f"Assistant: {response}\n"
    add_prompt_to_conversation(message.content, conversation_actual)
    conversation_actual.append({"role": "assistant", "content": response })
    end = time.time()
    total_time = end - start
    await cl.Message(f"Assistance: {response}").send()
    

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)