import os
import requests
import fitz
import pandas as pd
import re
import time
import numpy as np
import arxiv

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm
from spacy.lang.en import English
from vector_server import FaissKNNSearch
from os import listdir
from os.path import isfile, join
from typing import List

def split_list(
    input_list: List[str],
    slice_size: int=10
    ) -> List[List[str]]:
  return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def download_pdfs_from_arxiv(
        topic:str,
        data_path: str,
        num_results:int = 10
):
    client = arxiv.Client()
    search = arxiv.Search(
    query = topic,
    max_results = num_results,
    sort_by = arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)
    os.makedirs(data_path, exist_ok=True)
    print(f"[INFO] Downloading from arxiv...")
    for paper in tqdm(results):
        paper.download_pdf(data_path)

    
def download_single_pdf(
    url:str,
    data_path: str
):
    pdf_path = os.path.join(data_path, "file.pdf")
    if not os.path.exists(pdf_path):
        print("[INFO] File doesn't exist, downloading...")
        os.makedirs(data_path, exist_ok=True)

        response = requests.get(url)
        if response.status_code == 200:
            print(f"[INFO] downloaded {round(len(response.content)/(1024**2), 2)} MB, storing...")
            with open(pdf_path, "wb+") as file:
                file.write(response.content)
                print(f"[INFO] The file has been dowloaded and saved as {pdf_path}")
        else:
            raise Exception(f"[ERROR] Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"[INFO] file {pdf_path} already exists")

def text_formatter(
    text: str
) -> str:
  """Performs minor formatting on text"""
  cleaned_text = text.replace("\n", " ").strip()

  return cleaned_text

def split_list(
    input_list: List[str],
    slice_size: int
    ) -> List[List[str]]:
  return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def open_and_read_pdfs(
    data_path: str,
) -> List[dict]:

  pdf_files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
  pages_and_texts = []

  for pdf_path in pdf_files:
    if pdf_path.split(".")[-1] != "pdf":
        raise Exception(f"[ERROR] Found file {pdf_path} which is not a pdf")
      
    doc = fitz.open(pdf_path)
    
    for page in doc:
        text = page.get_text()
        text = text_formatter(text)

        pages_and_texts.append({
            "pdf_file": pdf_path,
            "page_number": page.number,
            "page_chat_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text)/4,
            "text": text
        })
  return pages_and_texts

def prepare_text(
    pages_and_texts: dict,
    num_sentence_chunk_size: int,
    min_token_length: int
) -> pd.DataFrame:

    nlp = English()
    nlp.add_pipe("sentencizer")
    print("[INFO] Splitting the text of the pages into sentences")
    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(val) for val in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    
    print(f"[INFO] Splitting the senteces into at most {num_sentence_chunk_size} chunks")
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(
            input_list = item["sentences"],
            slice_size = num_sentence_chunk_size
        )

        item["num_chunks"] = len(item["sentence_chunks"])

    pages_and_chunks = []

    print("[INFO] Merges the chunks to create a paragraph")
    for item in tqdm(pages_and_texts):
        for chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]

            joined_senteced_chunk = "".join(chunk).replace("  ", " ").strip()
            joined_senteced_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_senteced_chunk)
            chunk_dict["sentence_chunk"] = joined_senteced_chunk
            chunk_dict["chunk_char_count"] = len(joined_senteced_chunk)
            chunk_dict["chunk_token_count"] = len(joined_senteced_chunk)/4
            chunk_dict["chunk_word_count"] = len(joined_senteced_chunk.split(" "))
            pages_and_chunks.append(chunk_dict)

    print("[INFO] Some statistics about your pdf:")
    df = pd.DataFrame(pages_and_chunks)
    pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > min_token_length].to_dict(orient = "records")
    return pages_and_chunks_over_min_token_len

def create_and_store_embeddings(
    pages_and_chunks_over_min_token_len:pd.DataFrame,
    embedding_model_name:str,
    batch_size:int,
    csv_path:str,
    top_k_context:int,
    device:str = "cpu"
) -> (FaissKNNSearch, SentenceTransformer):

    data_path = join(csv_path, "text_chunks_and_embeddings_df.csv")
    print(f"[INFO] Loading {embedding_model_name} model on {device}...")
    embedding_model = SentenceTransformer(
        model_name_or_path = embedding_model_name,
        device = device
    ).to(device)

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
        text_chunks_and_emnbeding_df.to_csv(data_path, index=False)
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

    return faiss_dataset, embedding_model

def loading_re_ranking_model(
        re_ranking_model_name:str,
        device:str = "cpu"
    ) -> CrossEncoder:

    print(f"[INFO] Loading {re_ranking_model_name} model on {device}...")
    return CrossEncoder(
        re_ranking_model_name,
        device=device
    )

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
    