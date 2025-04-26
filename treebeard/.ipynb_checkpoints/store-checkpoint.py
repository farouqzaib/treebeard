import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import heapq

@dataclass
class Document:
    id: str
    content: str
    keywords: str
    embedding: np.ndarray
    source: str
    url: Optional[str] = None
    
    def get_citation(self):
        if self.url:
            return f"[Source: {self.source} - {self.url}]"
        return f"[Source: {self.source}]"

class VectorIndex:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        scores = [(doc, np.dot(query_embedding, doc.embedding) / 
                  (np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)))
                  for doc in self.documents]
        return [doc for doc, _ in heapq.nlargest(top_k, scores, key=lambda x: x[1])]    