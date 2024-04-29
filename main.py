from utils import (
    check_pdfs_presence,
    download_single_pdf, 
    download_pdfs_from_arxiv,
    prepare_text,
    open_and_read_pdfs,
    create_and_store_embeddings,
    loading_re_ranking_model,
    retrieve_topk_results,
)

from llm_utils import(
    get_device,
    clean_memory,
    load_llm,
    generate_reply
)

import argparse

def __pars_args__():
    parser = argparse.ArgumentParser(description='RAG')

    parser.add_argument(
        '--url', 
        type=str, 
        default=None, 
        help='Url to download the pdf file (Default None)'
    )

    parser.add_argument(
        '--data_path', 
        type=str, 
        default="./data", 
        help='Path where to download the data (Default ./data)'
    )

    parser.add_argument(
        '--csv_path', 
        type=str, 
        default="./csv", 
        help='Path where to download the csv (Default ./csv)'
    )

    parser.add_argument(
        '--num_sentence_chunk_size', 
        type=int, 
        default=10, 
        help='How many senteces form a chunk (Default 10)'
    )

    parser.add_argument(
        '--min_token_length', 
        type=int, 
        default=30, 
        help='Discard the paragraphs with less than X tokens (Default X=30)'
    )

    parser.add_argument(
        '--embedding_model_name', 
        type=str, 
        default="all-mpnet-base-v2", 
        help='Model name used to create embeddings (Default all-mpnet-base-v2)'
    )

    parser.add_argument(
        '--re_ranking_model_name', 
        type=str, 
        default="mixedbread-ai/mxbai-rerank-large-v1", 
        help='Model name used to re-rank information retrieved using the query (Default all-mpnet-base-v2)'
    )

    parser.add_argument(
        '--llm_model_name', 
        type=str, 
        default="google/gemma-1.1-2b-it", 
        help='LLM used for generation step (Default google/gemma-1.1-2b-it)'
    )

    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128, 
        help='Batch size used to crate the embeddings for the text (Default 128)'
    )

    parser.add_argument(
        '--top_k_context', 
        type=int, 
        default=5, 
        help='Number of items used as context by the LLM (Default 5)'
    )

    parser.add_argument(
        '--token', 
        type=str, 
        default=None, 
        help='Hugging face token needed to download models (Default None)'
    )

    parser.add_argument(
        '--num_results', 
        type=int, 
        default=10, 
        help='Number of papers to download from arvix. Valid only if URL is None (Default 10)'
    )

    parser.add_argument(
        '--low_cpu_mem_usage', 
        type=bool, 
        default=False, 
        help='(Default False)'
    )
    
    args = parser.parse_args()
    print("".join(["-"]*10), "\n")
    for argument, val in vars(args).items():
        print(argument, "\t", val)
    print("".join(["-"]*10), "\n")
    return args
    

def main(args):
    device, attn_implementation, quantization_config = get_device()
    if check_pdfs_presence(args.data_path):
        print(f"[INFO] Files in {args.data_path} found, skipping download step...")
    elif(args.url is None):
        print("Insert argument: ")
        topic = input()
        download_pdfs_from_arxiv(
            topic = topic,
            data_path = args.data_path,
            num_results = args.num_results
        )
    else:
        download_single_pdf(
            url = args.url,
            data_path = args.data_path
        )
        

    pages_and_texts = open_and_read_pdfs(
        data_path = args.data_path,
    )

    pages_and_chunks_over_min_token_len = prepare_text(
        pages_and_texts = pages_and_texts,
        num_sentence_chunk_size = args.num_sentence_chunk_size,
        min_token_length = args.min_token_length
    )

    faiss_dataset, embedding_model = create_and_store_embeddings(
        pages_and_chunks_over_min_token_len = pages_and_chunks_over_min_token_len,
        embedding_model_name = args.embedding_model_name,
        batch_size = args.batch_size,
        csv_path = args.csv_path,
        top_k_context = args.top_k_context,
        device = device
    )

    re_ranking_model = loading_re_ranking_model(
        re_ranking_model_name = args.re_ranking_model_name,
        device = device
    )

    tokenizer, llm_model = load_llm(
        llm_model_name = args.llm_model_name,
        attn_implementation = attn_implementation,
        low_cpu_mem_usage = args.low_cpu_mem_usage,
        quantization_config = quantization_config,
        token = args.token,
        device = device
    )


    while True:
        print("Insert query:")
        query = input()
        if query == "quit program":
            break
        selected_text, selected_pages = retrieve_topk_results(
            query = query,
            faiss_dataset = faiss_dataset,
            embedding_model = embedding_model,
            re_ranking_model = re_ranking_model,
            top_k_context = args.top_k_context
        )

        outputs_decoded = generate_reply(
            llm_model = llm_model,
            tokenizer = tokenizer,
            query = query,
            selected_text = selected_text,
            selected_pages = selected_pages,
            device = device
        )

        print(f"{outputs_decoded}")

        clean_memory(
            device = device
        )

if __name__ == "__main__":
    args = __pars_args__()
    main(args)