import torch
import numpy as np
import subprocess
import sys

if torch.cuda.is_available():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
else:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])

import faiss

class FaissKNNSearch:
  def __init__(
      self,
      k:int,
      device:str,
  ):
    self.k = k
    self.device = device

  def fit(
      self,
      embeddings: np.array,
      text: np.array,
      pages: np.array
  ):

    self.index = faiss.IndexFlatIP(embeddings.shape[1])
    if self.device == "cuda":
      self.index = faiss.index_cpu_to_all_gpus(self.index)
    
    self.index.add(embeddings.astype(np.float32))
    self.text = text
    self.pages = pages

  def retrieve(
      self,
      query: str
  )-> (np.array, np.array, np.array):

    _, indices = self.index.search(
        query.astype(np.float32),
        k = self.k
    )
    indices = indices[0]
    return indices, self.text[indices], self.pages[indices]
    