from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BitsAndBytesConfig
import numpy as np
import torch
from tqdm.auto import tqdm
import json
import time
import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
login(token = os.environ['hf_token'])

model_id = "Viet-Mistral/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype = torch.float16,
    device_map = "auto",
    use_cache = True,
)