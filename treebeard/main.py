import pandas as pd
import argparse
from treebeard.store import Document, VectorIndex
from treebeard.crawler import WebSearch
from treebeard.research import MCTSDeepResearch
from treebeard.generator import text_generator, embedding_generator
import os

def main(query, csv_path, generate_outline_from_archive):
    documents = pd.read_csv(csv_path)

    docs = []
    for i in range(documents.shape[0]):
        row = documents.iloc[i]
        docs.append(Document(
            row['_id'], 
            row['text'], 
            row['topics'],
            embedding_generator(row['text']), 
            row['filename']
        ))

    print(f"Loaded {len(docs)} documents.")
    vector_index = VectorIndex(docs)

    web_search = WebSearch(os.environ['GOOGLE_API_KEY'], os.environ['CUSTOM_SEARCH_ENGINE_ID'])
    assistant = MCTSDeepResearch(
        vector_index, 
        web_search,
        text_generator, 
        embedding_generator,
        max_iterations=50
    )

    outline, path, searches = assistant.generate_research_report(
        query=query,
        generateOutlineFromArchive=generate_outline_from_archive
    )

    report = assistant.generate_final_document(outline)

    output_path = f"examples/{query}_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True, help='Query for the report')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--generate_outline_from_archive', action='store_true', help='Generate outline from archive if set')
    args = parser.parse_args()

    main(args.query, args.csv_path, args.generate_outline_from_archive)
