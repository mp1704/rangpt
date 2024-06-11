from fastapi import FastAPI
import uvicorn
import time
from handle_conversation import *
# from typing import Optional, List

app = FastAPI()

@app.get("/")
def root():
    return {"message": "hello world"}

@app.post("/qa")
def search_qa(user_input: str):
    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
    
    conversation_actual = [{"role": "system", "content": system_prompt }]  
    conversation_history = ""
    
    while True:
        # user_input = input("User: ")
        start = time.time()
        conversation_history += f"User: {user_input}\n"
        conversation_actual.append({"role": "user", "content": user_input })
        
        if user_input.lower() == "quit":
            # print("Assistant: Tạm biệt!")
            # break
            return {"Assistant": "Tạm biệt!"}

        response_conv_turn = handle_conversation_turn(conversation_history, conversation_actual)
        response = response_conv_turn
        conversation_history += f"Assistant: {response}\n"
        add_prompt_to_conversation(user_input, conversation_actual)
        conversation_actual.append({"role": "assistant", "content": response })
        
        end = time.time()
        total_time = end - start
        # print(f"Assistant: {response}\n")
        # print(f"Time: {total_time:.2f} seconds")
        return {"Assistant": response, "request_time": f"Time: {total_time:.2f}"}
    

if __name__ == "__main__":
    uvicorn.run("qa_app:app", host="0.0.0.0", port=8000)