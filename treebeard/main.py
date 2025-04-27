import pandas as pd
from treebeard.store import Document, VectorIndex
from treebeard.crawler import WebSearch
from treebeard.research import MCTSDeepResearch
from treebeard.generator import text_generator, embedding_generator
import os

def main():

    documents = pd.read_csv('/home/farouqzaib/Downloads/archiving_1970.csv')

    docs = []

    for i in range(documents.shape[0]):
        row = documents.iloc[i]

        docs.append(Document(row['_id'], row['text'], row['topics'],
                            embedding_generator(row['text']), row['filename']))
                
    print(len(docs))
    vector_index = VectorIndex(docs)

    web_search = WebSearch(os.environ['GOOGLE_API_KEY'], os.environ['CUSTOM_SEARCH_ENGINE_ID'])
    assistant = MCTSDeepResearch(
        vector_index, 
        web_search,
        text_generator, 
        embedding_generator,
        max_iterations=50
    )

    outline, path, searches = assistant.generate_research_report("Economic development of Nigeria contrasting 1970 with current trends",
                                                                 outlineFromArchive=False)

    # Generate final document
    report = assistant.generate_final_document(outline)

    print(f"Report: {outline.title}")
    print(f"Used {searches} web searches")
    print(report)
    print("=" * 50)
    print(f"Overall confidence: {outline.overall_confidence:.2f}")
    print(f"Citations: {outline.citation_count}")

    # return document, outline, path


if __name__ == "__main__":
    main()