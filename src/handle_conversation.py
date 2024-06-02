from typing import List, Dict
import sys
import logging
logging.basicConfig(level=logging.INFO)
from init_component import *
from llm import *

# TODO: do argument parsing
top_k_retrieve = 10
top_k_rerank = 5
collection_name = 'rangpt'

def chat(user_input: str):
    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
    
    conversation_actual = [{"role": "system", "content": system_prompt }]  
    conversation_history = ""
    
    while True:
        user_input = input("User: ")
        start = time.time()
        conversation_history += f"User: {user_input}\n"
        conversation_actual.append({"role": "user", "content": user_input })
        
        if user_input.lower() == "quit":
            print("Assistant: Tạm biệt!")
            break

        response_conv_turn = handle_conversation_turn(conversation_history, conversation_actual)
        response = response_conv_turn
        conversation_history += f"Assistant: {response}\n"
        add_prompt_to_conversation(user_input, conversation_actual)
        conversation_actual.append({"role": "assistant", "content": response })
        
        end = time.time()
        total_time = end - start
        print(f"Assistant: {response}\n")
        print(f"Time: {total_time:.2f} seconds")
        
    return conversation_history

def handle_conversation_turn(conversation_history:str , conversation_actual: List[Dict]):
    # prompt của query_reformulation: nếu là câu hỏi thì reformulate câu hỏi, nếu không thì giữ nguyên
    reformulated_query = query_reformulation(conversation_history, conversation_actual)
    
    logging.info(f"{reformulated_query = }")
    method_for_answering = choose_method_for_handling_user_query(reformulated_query, conversation_actual)

    
    if "A." in method_for_answering or "A" == method_for_answering:
        print("\n--------\nLOGGING: A.Trả lời trực tiếp câu hỏi. Câu hỏi đã được diễn đạt lại: ", reformulated_query, "\n--------")
        return answer_user_directly(reformulated_query, conversation_actual)

    elif "B." in method_for_answering or "B" == method_for_answering:
        # old approach is finding method first, if B then reformulate query
        # reformulated_query = query_reformulation(conversation_history, conversation_actual)
        print("\n--------\nLOGGING: B.Trả lời câu hỏi bằng cách tìm kiếm thông tin. Câu hỏi đã được diễn đạt lại: ", reformulated_query, "\n--------")
        query_results = client.search(
            collection_name=collection_name,
            query_vector=SENTEMB.encode(reformulated_query)['dense_vecs'].tolist(),
            limit=top_k_retrieve,
        )

        raw_contexts = []
        for query_result in query_results:
            raw_contexts.append(query_result.payload['text'])
        rerank_scores = reranker.compute_score([[reformulated_query, raw_contexts[i]] for i in range(len(raw_contexts))], normalize=True)
        rerank_scores = np.array(rerank_scores)

        top_k_rerank_indices = rerank_scores.argsort()[-len(raw_contexts):][::-1]

        reranked_contexts = [raw_contexts[top_k_rerank_indices[i]] for i in range(len(raw_contexts))][:top_k_rerank]
        return answer_query_with_context(reformulated_query, conversation_history, reranked_contexts, conversation_actual)

    elif "C." in method_for_answering or "C" == method_for_answering:
        print("\n--------\nLOGGING: C.Yêu cầu làm rõ câu hỏi. Câu hỏi đã được diễn đạt lại: ", reformulated_query, "\n--------")
        return ask_for_clarification_questions(conversation_history, conversation_actual)
    else:
        print("\n--------\nLOGGING: Không có chữ cái A, B, C trong câu trả lời \n--------")

def answer_user_directly(reformulated_query, conversation_actual):
    prompt = f"""Bạn là trợ lý ảo lịch sự, thông minh.

<CÂU HỎI>: {reformulated_query}

Vui lòng trả lời một cách trung thực và hữu ích
Trợ lý ảo:
    """
    add_prompt_to_conversation(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant

def query_reformulation(conversation_history, conversation_actual):
    user_last_query = separate_last_user_query(conversation_history)
    prompt = f"""Cho đoạn hội thoại sau:
<HỘI THOẠI>: {conversation_history}
<TRUY VẤN CUỐI CÙNG>: {user_last_query}
Nếu truy vấn của người dùng là một câu hỏi thì hãy viết lại câu truy vấn đó để câu truy vấn viết lại đó có thể dùng để tìm kiếm và thỏa mãn nhu cầu thông tin của người dùng. Tốt nhất câu truy vấn được sửa lại ở dạng câu hỏi. Nếu đó không phải là câu hỏi thì giữ nguyên.
Truy vấn của người dùng viết lại là:
    """
    add_prompt_to_conversation(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant
    
def choose_method_for_handling_user_query(reformulated_query, conversation_actual):
    prompt = f"""Bạn là trợ lý ảo lịch sự, thông minh và bạn có cơ sở dữ liệu thông tin về các loài rắn

<YỀU CẦU>: {reformulated_query}

Để trả lời câu hỏi của người dùng một cách hữu ích và chính xác, bạn hãy lựa chọn một trong các phương án sau để trả lời: 

A. Trả lời trực tiếp câu hỏi (áp dụng cho lời chào, câu hỏi đơn giản, thông thường sẽ không liên quan tới rắn)
B. Lấy thông tin từ cơ sở dữ liệu và trả lời dựa trên đó
C. Yêu cầu làm rõ câu hỏi (chỉ được thực hiện khi bạn không hiểu gì hoặc câu hỏi bị vô nghĩa)

Lưu ý: Bắt buộc trả lời sử dụng A. hoặc B. hoặc C.
Đáp án của bạn là
 """

    add_prompt_to_conversation(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant


def join_list_into_string(my_list):
    result = ""
    for i, item in enumerate(my_list):
        result += f"{i+1}. {item}\n\n"
    return result.rstrip()

def check_if_context_is_relevant(query, string_context, conversation_actual):
    prompt = f"""Cho truy vấn và nội dung sau đây. Kiểm tra xem nội dung có thông tin liên quan để trả lời truy vấn không?
<TRUY VẤN>: {query}

<NỘI DUNG>: {string_context}

Chọn:
A. Có
B. Không
Lưu ý: Trả lời sử dụng chữ cái A. hoặc B.
"""
    add_prompt_to_conversation(prompt, conversation_actual)
    assistant = generate(conversation_actual)
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
        
        logging.info("Answer query with following context:".upper())
        logging.info(string_context)
        logging.info("~"*70)
        
        prompt = f"""Bạn là trợ lý ảo lịch sự, thông minh
Dựa trên truy vấn của người dùng và nội dung truy xuất từ cơ sở dữ liệu.

Truy vấn được định dạng lại:
{query}

Nội dung được truy xuất:
{string_context}

Hãy trả lời người dùng dựa trên thông tin trong nội dung. Hãy bắt đầu câu trả lời bằng: "Dựa trên hiểu biết của tôi".
        """
        add_prompt_to_conversation(prompt, conversation_actual)
        assistant = generate(conversation_actual)
        return assistant

def separate_last_user_query(conversation):
    # str? 'User: xin chào\n' 
    lines = conversation.splitlines()

    if not lines:
        return None

    user_queries = [line.split(": ", maxsplit=1)[1] for line in lines if line.startswith("User:")]

    return user_queries[-1] if user_queries else None

# TODO: more options to generate
def generate(conversation: List[Dict]):
    """
    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
    tok = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = tok['input_ids']
    attention_mask = tok['attention_mask']
    with torch.no_grad():
        output_ids = model.generate(
            input_ids = input_ids,
            max_length = 32000,
            do_sample = True,
            attention_mask=attention_mask,
            top_p = 1.0,
            top_k = 4,
            temperature = 0.5,
            repetition_penalty = 1.0,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            use_cache = True,
        )
    assistant = tokenizer.batch_decode(output_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
    return assistant 
    """
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids = input_ids,
            max_length = 32000,
            do_sample = True,
            top_p = 1.0,
            top_k = 4,
            temperature = 0.5,
            repetition_penalty = 1.0,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            use_cache = True,
        )
    assistant = tokenizer.batch_decode(output_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
    return assistant
    
def add_prompt_to_conversation(prompt:str, conversation_actual: List[Dict]):
    """ 
    prompt will be reformulated, this function will replace the reformulated prompt to content of the last conversation turn
    """
    conversation_actual[-1]["content"] = prompt
    
def ask_for_clarification_questions(conversation_text:str, conversation_actual:List[Dict]):
    prompt = f"""Cho đoạn hội thoại sau
{conversation_text}

Tạo một câu hỏi làm rõ dựa trên lịch sử cuộc trò chuyện, với mục đích hữu ích nhất có thể cho người dùng
    """
    add_prompt_to_conversation(prompt, conversation_actual)
    assistant = generate(conversation_actual)
    return assistant

def DummyResponse():
    text = "Tôi rất xin lỗi vì sự bất tiện này, tôi không thể tìm thấy thông tin phù hợp cho câu hỏi của bạn"
    return text