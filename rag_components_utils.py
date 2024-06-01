from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
from vector_server import FaissKNNSearch
from os.path import join
from tqdm.auto import tqdm
import gc
import torch
import numpy as np
import pandas as pd
import time
import os

def get_device() -> (str, str, BitsAndBytesConfig):

    quantization_config = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device available: {device}")
    if device == "cuda":
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2**30))
        print(f"[INFO] Available GPU memory: {gpu_memory_gb} GB")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

    if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    print(f"[INFO] Using attn_implementation: {attn_implementation}")
    print(f"[INFO] Quantization config used: {quantization_config}")
    return device, attn_implementation, quantization_config

def clean_memory(
        device:str
    ):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

def load_llm(
        llm_model_name:str,
        attn_implementation:str,
        low_cpu_mem_usage:bool,
        quantization_config:BitsAndBytesConfig = None,
        token:str = None,
        device:str = "cpu"
) -> (AutoTokenizer, torch.nn.Module):  
    
    print(f"[INFO] Loading tokenizer and model for {llm_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = llm_model_name,
        token = token
    )
    if quantization_config is not None:
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = llm_model_name,
            token = token,
            low_cpu_mem_usage = low_cpu_mem_usage,
            attn_implementation = attn_implementation,
            quantization_config = quantization_config
        )
    else: 
        llm_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path = llm_model_name,
            token = token,
            low_cpu_mem_usage = low_cpu_mem_usage,
            attn_implementation = attn_implementation,
        )

    num_params = sum([param.numel() for param in llm_model.parameters()])
    mem_params = sum([param.numel() * param.element_size() for param in llm_model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in llm_model.parameters()])

    model_mem_bytes = mem_params + mem_buffers
    model_mem_gb = round(model_mem_bytes/(1024**3), 2)

    print(f"[INFO] Using {llm_model_name} on {device} with {num_params} parameters loaded on {model_mem_gb} GB memory")

    return tokenizer, llm_model

def load_re_ranking_model(
        re_ranking_model_name:str,
        device:str = "cpu"
    ) -> CrossEncoder:

    print(f"[INFO] Loading {re_ranking_model_name} model on {device}...")
    return CrossEncoder(
        re_ranking_model_name,
        device=device
    )

def load_embedding_model(
        embedding_model_name: str,
        device: str
    ) -> SentenceTransformer:
    print(f"[INFO] Loading {embedding_model_name} model on {device}...")
    return SentenceTransformer(
            model_name_or_path = embedding_model_name,
            device = device
        )

def create_and_store_embeddings(
    pages_and_chunks_over_min_token_len:pd.DataFrame,
    embedding_model:SentenceTransformer,
    batch_size:int,
    csv_path:str,
    top_k_context:int,
    device:str = "cpu"
) -> FaissKNNSearch:
    
    os.makedirs(csv_path, exist_ok=True)
    data_path = join(csv_path, "text_chunks_and_embeddings_df.csv")
    
    if not os.path.exists(data_path):
        
        print("[INFO] Embedding creation started...")
        start_time=time.time()
        embedding_vector = embedding_model.encode(
                pd.DataFrame(pages_and_chunks_over_min_token_len)["sentence_chunk"].to_numpy(),
                batch_size = batch_size,
                convert_to_tensor=True
        )
        end_time=time.time()
        print(f"[INFO] Total time required to create the embeddings: {end_time - start_time}")
        
        for item, vector in tqdm(zip(pages_and_chunks_over_min_token_len, embedding_vector.detach().cpu().numpy())):
            item["embedding"] = vector
        text_chunks_and_emnbeding_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
        print("[INFO] Saving embeddings...")
        text_chunks_and_emnbeding_df.to_csv(data_path, index=False, escapechar="\\")
        print(f"[INFO] Embedding stored at: {data_path}")
    else:
        print(f"[INFO] File {data_path} alerady exists, loading it...")

    data = pd.read_csv(data_path)
    embeddings = data["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep="  ").astype(np.float32))
    embeddings = np.stack(embeddings)

    print("[INFO] Creating faiss dataset...")
    
    faiss_dataset = FaissKNNSearch(
        k=top_k_context * 3,
        device=device
    )
    faiss_dataset.fit(
        embeddings = embeddings,
        text = data["sentence_chunk"].to_numpy(),
        pages = data["page_number"].to_numpy()
    )

    return faiss_dataset
def retrieve_topk_results(
    query:str,
    faiss_dataset:FaissKNNSearch,
    embedding_model:SentenceTransformer,
    re_ranking_model:CrossEncoder,
    top_k_context:int
) -> (np.array, np.array):
    
    query_embedding = embedding_model.encode(query, convert_to_tensor = True).detach().cpu().view(1, -1).numpy()
    _, text, pages = faiss_dataset.retrieve(query_embedding)

    re_ranked_result = re_ranking_model.rank(
        query,
        text,
        top_k=top_k_context
    )

    idx = pd.DataFrame(re_ranked_result)["corpus_id"].to_numpy()
    selected_text = text[idx]
    selected_pages = pages[idx]

    return selected_text, selected_pages

def generate_reply(
    llm_model:torch.nn.Module,
    tokenizer:AutoTokenizer,
    query:str,
    selected_text:np.array,
    device:str = "cpu"
) -> str:
    
    context = "- " + "\n- ".join(selected_text)
    base_prompt = f"""Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Do not return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following context items to answer query:
{context}

User query: {query}
Answer:
    """

    dialoge_template = [
        {
            "role": "user",
            "content": base_prompt,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        conversation = dialoge_template,
        tokenize=False,
        add_generation_prompt=True
    )

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    outputs = llm_model.generate(
        **input_ids,
        temperature = 0.7,
        do_sample = True,
        max_new_tokens = 128
    )

    outputs_decoded = tokenizer.decode(
        outputs[0], 
        skip_special_tokens = False
    )

    for code in ["/s", "bos", "eos"]:
        outputs_decoded = outputs_decoded.strip(f"<{code}>").replace(prompt.strip(f"<{code}>"), "")

    return outputs_decoded.strip(" ")
    