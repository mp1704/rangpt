
from llama_cpp import Llama

model_args = {
      "max_tokens": 32000,
      "top_p": 1.0,
      "top_k": 4,
      "temperature": 0.5,
      "repetition_penalty": 1.0,
    #   "f16_kv": True,
}
llm = Llama(model_path="./ggml-vistral-7B-chat-q8.gguf",
            n_gpu_layers=-1,
            n_ctx= 512*5,
            **model_args)


from typing import List, Dict

def handle_conversation_turn(conversation_history, conversation_actual):
    method_for_answering = choose_method_for_handling_user_query(conversation_history, conversation_actual)

    if "A." in method_for_answering or "A" == method_for_answering:
        print("\n--------\nLOGGING: Trả lời trực tiếp câu hỏi \n--------\n")
        return answer_user_directly(conversation_history, conversation_actual)

    elif "B." in method_for_answering or "B" == method_for_answering:
        reformulated_query = query_reformulation(conversation_history, conversation_actual)
        print("\n--------\nLOGGING: Trả lời câu hỏi bằng cách tìm kiếm thông tin. Câu hỏi đã được diễn đạt lại: ", reformulated_query, "\n--------\n")
        # query_results = client.search(
        #     collection_name="rangpt",
        #     query_vector=SENTEMB.encode(reformulated_query)['dense_vecs'].tolist(),
        #     limit=3,
        # )
        from rag import qdrant_search
        query_results = qdrant_search(collection_name="rangpt",
                                    reformulated_query=reformulated_query,
                                    limit=3,
                                    )

        contexts = []
        results = []
        for query_result in query_results:
            result = query_result.payload
            results.append(result)
        
            
        for d in results:
          context = "\n".join(f"{key}: {value}" for key, value in d.items())
          contexts.append(context)
        return answer_query_with_context(reformulated_query, conversation_history, contexts,conversation_actual)

    elif "C." in method_for_answering or "C" == method_for_answering:
        print("\n--------\nLOGGING: Yêu cầu làm rõ câu hỏi \n--------\n")
        return ask_for_clarification_questions(conversation_history, conversation_actual)
    else:
        print("\n--------\nLOGGING: Không có chữ cái A, B, C trong câu trả lời \n--------\n")

def answer_user_directly(conversation_history, conversation_actual):
    prompt = f"""Bạn là trợ lý ảo lịch sự, thông minh.
    Cho đoạn hội thoại sau:
{conversation_history}

Vui lòng trả lời một cách trung thực và hữu ích
Trợ lý ảo:
    """
    replace_string(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant

def query_reformulation(conversation_history, conversation_actual):
    user_last_query = separate_last_user_query(conversation_history)
    prompt = f"""Cho đoạn hội thoại sau:
{conversation_history}

Truy vấn cuối cùng của người dùng:
{user_last_query}

Xin hãy viết lại câu truy vấn cuối của người dùng để câu truy vấn viết lại đó có thể dùng để tìm kiếm và thỏa mãn nhu cầu thông tin của người dùng. Tốt nhất câu truy vấn được sửa lại ở dạng câu hỏi
    """
#     print('conversation_actual: ',conversation_actual)
    replace_string(prompt, conversation_actual)
#     conversation_actual.append({"role": "user", "content": prompt})
    assistant = generate(conversation_actual)
#     print('assistant', assistant)
    return assistant
    
def choose_method_for_handling_user_query(conversation_history, conversation_actual):
#     print('conversation_history: ', conversation_history)
    user_last_query = separate_last_user_query(conversation_history)
#     print('user_last_query: ',user_last_query)
    prompt = f"""Bạn là trợ lý ảo lịch sự, thông minh và bạn có cơ sở dữ liệu thông tin về các loài rắn
Cho đoạn hội thoại sau:
{conversation_history}

Truy vấn cuối cùng của người dùng:
{user_last_query}

Để trả lời câu hỏi cuối cùng của người dùng một cách hữu ích và chính xác, bạn hãy lựa chọn một trong các phương án sau để trả lời: 

A. Trả lời trực tiếp câu hỏi mà bạn đã biết cách trả lời (áp dụng cho lời chào, cuộc trò chuyện thông thường, v.v.)
B. Lấy thông tin từ cơ sở dữ liệu và trả lời dựa trên đó
C. Yêu cầu làm rõ câu hỏi (Chỉ yêu cầu làm rõ khi cần thiết)

Lưu ý: Bắt buộc trả lời sử dụng A. hoặc B. hoặc C. 
 """
    replace_string(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant

def join_list_into_string(my_list):
    result = ""
    for i, item in enumerate(my_list):
        result += f"{i+1}. {item}\n\n"
    return result.rstrip()

def check_if_context_is_relevant(query, string_context, conversation_actual):
    prompt = f"""Cho truy vấn và nội dung sau đây. Kiểm tra xem nội dung có thông tin liên quan để trả lời truy vấn không?
Truy vấn: {query}

Nội dung: {string_context}

Chọn:
A. Có
B. Không
Lưu ý: Trả lời sử dụng chữ cái A. hoặc B.
"""
#     print('prompt for check relevant: ', prompt)
    replace_string(prompt, conversation_actual)
    assistant = generate(conversation_actual)
#     print("assisatnt for check revelant: ", assistant)
    if "A." in assistant or "A" == assistant: 
        return True
    else: 
        return False

def answer_query_with_context(query, conversation_text, contexts, conversation_actual):
    string_context = join_list_into_string(contexts)

    context_relevant = check_if_context_is_relevant(query, string_context, conversation_actual)

    if not context_relevant: 
        return DummyResponse()

    else:
        print(f"\n--------\nMESSAGE: {string_context}\n--------\n")
        prompt = f"""Bạn là trợ lý ảo lịch sự, thông minh
Dựa trên lịch sử cuộc trò chuyện sau đây, bối cảnh và truy vấn của người dùng đã được định dạng lại.

Lịch sử cuộc hội thoại:
{conversation_text}

Truy vấn được định dạng lại:
{query}

Nội dung được truy xuất:
{string_context}

Hãy trả lời người dùng dựa trên thông tin trong nội dung
        """
        replace_string(prompt, conversation_actual)
        assistant = generate(conversation_actual)
        return assistant

def separate_last_user_query(conversation):
    lines = conversation.splitlines()

    if not lines:
        return None

    user_queries = [line.split(": ", maxsplit=1)[1] for line in lines if line.startswith("User:")]

    return user_queries[-1] if user_queries else None

def ask_for_clarification_questions(conversation_history, conversation_actual):
    prompt = f"""Cho đoạn hội thoại sau
{conversation_history}

Tạo một câu hỏi làm rõ dựa trên lịch sử cuộc trò chuyện, với mục đích hữu ích nhất có thể cho người dùng
    """
    replace_string(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant

def generate(conversation: List[Dict]):
    output = llm.create_chat_completion(
        messages=[
            *conversation
            # {
            #     "role": "system",
            #     "content": system_prompt,
            # },
            # {"role": "user", "content": conversation[-1]['content']},
        ]
    )
    return output['choices'][0]['message']['content']
   
    
def replace_string(string_to_replace, conversation_actual):
    # Modify the last dictionary in the list
    conversation_actual[-1]["content"] = string_to_replace
    
def ask_for_clarification_questions(conversation_text, conversation_actual):
    prompt = f"""Cho đoạn hội thoại sau
{conversation_text}

Tạo một câu hỏi làm rõ dựa trên lịch sử cuộc trò chuyện, với mục đích hữu ích nhất có thể cho người dùng
    """
    replace_string(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant

def DummyResponse():
    text = "Tôi rất xin lỗi vì sự bất tiện này, tôi không thể tìm thấy thông tin phù hợp cho câu hỏi của bạn"
    return text