import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import List
from treebeard.store import Document

class WebSearch:
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.search_history = []

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        self.search_history.append(query)
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": top_k
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json()

        docs = []
        for i, item in enumerate(results.get("items", [])):
            doc_id = f"web_{len(self.search_history)}_{i}"
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")

            # Try fetching and parsing the full page content
            try:
                page_response = requests.get(link, timeout=5)
                page_response.raise_for_status()
                soup = BeautifulSoup(page_response.text, "html.parser")

                # Remove scripts/styles and extract text
                for script_or_style in soup(["script", "style", "noscript"]):
                    script_or_style.decompose()
                full_text = soup.get_text(separator=" ", strip=True)

                content = f"{title}: {full_text[:2000]}"#only first 2k chars. Do something smarter next go around
            except Exception as e:
                content = f"{title}: {snippet} (Failed to fetch full page: {e})"

            embedding = np.random.rand(768)  # Placeholder
            docs.append(Document(doc_id, content, "", embedding, "Google Search", link))

        return docs