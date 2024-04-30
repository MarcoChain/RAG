from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
import gc
import torch
import numpy as np

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

def generate_reply(
    llm_model:torch.nn.Module,
    tokenizer:AutoTokenizer,
    query:str,
    selected_text:np.array,
    selected_pages:np.array,
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