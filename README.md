# RANGPT
**in progress**
## Demo
### Chatbot with Chainlit

![cl_demo](assets/images/cl_demo_img.png)

### Single request with FastAPI
![qa demo](assets/images/fastapi_demo_img.png)

## How to use 

- Create a virtual environment and install the required packages
```
conda create -n rangpt python=3.10
conda activate rangpt
pip install -r requirements.txt
```
- Go to folder `src`, create a `.env` file then add Qdrant API key and HuggingFace token (make sure you have acces to *Viet-Mistral/Vistral-7B-Chat*) with the following format:
```
hf_token = "YOUR_HF_TOKEN"
qdrant_api_key = "YOUR_QDRANT_API_KEY"
```
- if you want to use the chatbot with Chainlit, run
```
chainlit run cl.py
```
or 
```
python cl.py
```
![quick_tutorial](assets/gif/demo_cl.gif)
- if you want to use the chatbot with FastAPI, run
```
python app.py
```
then go to `localhost:8000/docs` to test the API


## llama.cpp (in progress)
- download the model from [here](https://huggingface.co/uonlp/Vistral-7B-Chat-gguf/tree/main) and run `main.py`
```
!wget https://huggingface.co/uonlp/Vistral-7B-Chat-gguf/resolve/main/ggml-vistral-7B-chat-q4_0.gguf
!mv ggml-vistral-7B-chat-q4_0.gguf model/
!python main.py
```

