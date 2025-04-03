import os
import re

import pandas as pd
from datasets import load_dataset
from huggingface_hub import InferenceClient, notebook_login
from tqdm.auto import tqdm

# os.environ['HTTP_PROXY'] = "http://l00841179:15350068547Aa%3F@proxyhk.huawei.com:8080/"
# os.environ['HTTPS_PROXY'] = "http://l00841179:15350068547Aa%3F@proxyhk.huawei.com:8080/"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 镜像站加速
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
tqdm.pandas()  # load tqdm's pandas support
pd.set_option("display.max_colwidth", None)

notebook_login()
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
    token="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"
    # base_url='http://api.openai.rnd.huawei.com/v1',
    # api_key='sk-1234'
)

# Test your LLM client
llm_client.text_generation(prompt="How are you today?", max_new_tokens=20)