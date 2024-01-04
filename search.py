from typing import List, Tuple
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset 
from datasets.download import DownloadManager

if not torch.cuda.is_available():
    raise EnvironmentError('need CUDA env.')

class TextSearcher:
    INDEX_NAME = "multilingual-e5-large-passage/index_IVF2048_PQ256.faiss"
    WIKIPEDIA_JA_EMB_DS = "hotchpotch/wikipedia-passages-jawiki-embeddings"
    WIKIPEDIA_JA_DS_NAME = "passages-c400-jawiki-20230403"
    WIKIPEDIA_JA_DS = "singletongue/wikipedia-utils"
    MAX_SEQ_LENGTH = 512


    def __init__(self) -> None:
        self.dataset = load_dataset(path=self.WIKIPEDIA_JA_DS, name=self.WIKIPEDIA_JA_DS_NAME, split="train")
        
        self.emb_model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
        self.emb_model.max_seq_length = self.MAX_SEQ_LENGTH

        target_path = f"faiss_indexes/{self.WIKIPEDIA_JA_DS_NAME}/{self.INDEX_NAME}"
        dm = DownloadManager()
        index_local_path = dm.download(
            f"https://huggingface.co/datasets/{self.WIKIPEDIA_JA_EMB_DS}/resolve/main/{target_path}"
        )
        self.faiss_index = faiss.read_index(index_local_path)
        self.faiss_index.nprobe = 128


    def text_to_emb(self, text: str, prefix: str) -> np.ndarray:
        return self.emb_model.encode([prefix + text], normalize_embeddings=True)


    def search(self, question: str, top_k: int, search_text_prefix: str = "query") -> Tuple[List[Tuple[float, dict]], float, float]:
        emb = self.text_to_emb(question, search_text_prefix)
        scores, indexes = self.faiss_index.search(emb, top_k)
        scores = scores[0]
        indexes = indexes[0]
        results = []
        for idx, score in zip(indexes, scores):
            idx = int(idx)
            passage = self.dataset[idx]
            results.append((score, passage))
        return results
    
    def to_contexts(self, passages: List[dict]) -> str:
        contexts = ""
        for passage in passages:
            title = passage["title"]
            text = passage["text"]
            contexts += f"- {title}: {text}\n"
        return contexts


def create_text_searcher():
    return TextSearcher()

