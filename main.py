from utils import (
    download_single_pdf, 
    prepare_text,
    open_and_read_pdf,
    create_and_store_embeddings,
    loading_re_ranking_model,
    retrieve_topk_results
)

import argparse
import torch

def __pars_args__():
    parser = argparse.ArgumentParser(description='RAG')
    parser.add_argument(
        '--url', 
        type=str, 
        default="https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf", 
        help='Url to download the pdf file (Default Hawaii pressbook book url)'
    )

    parser.add_argument(
        '--data_path', 
        type=str, 
        default="./data", 
        help='Path where to download the data (Default ./data)'
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
    

    parser.add_argument('--task_id', type=int, default=20, help="Task to execute.")
    return parser.parse_args()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pdf_path = download_single_pdf(
        url = args.url,
        data_path = args.data_path
    )

    pages_and_texts = open_and_read_pdf(
        pdf_path = pdf_path,
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
        dest_path = args.data_path,
        top_k_context = args.top_k_context,
        device = device
    )

    re_ranking_model = loading_re_ranking_model(
        re_ranking_model_name = args.re_ranking_model_name,
        device = device
    )

    while True:
        print("Insert query:")
        query = input()
        if query == "quit program":
            break
        retrieve_topk_results(
            query = query,
            faiss_dataset = faiss_dataset,
            embedding_model = embedding_model,
            re_ranking_model = re_ranking_model,
            top_k_context = args.top_k_context,
            device = device
        )

if __name__ == "__main__":
    args = __pars_args__()
    main(args)