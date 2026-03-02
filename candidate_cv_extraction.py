import os
import pandas as pd
from pprint import pprint
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

CANDIDATE_DATA = r"outputs/fast_pdf_metadata.csv"


def fast_process_pdfs(directory_path: str,
                      output_filename: str = CANDIDATE_DATA) -> dict:
    """
    Fast PDF extraction pipeline.

    Loads all PDFs from a directory, extracts text + metadata,
    merges pages per document, and saves to CSV.

    Returns:
        dict -> status, file_path, total_pdfs
    """

    print(f"\n--- Starting Fast Extraction from: {directory_path} ---")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    loader = DirectoryLoader(
        directory_path,
        glob="*.pdf",
        loader_cls=PyMuPDFLoader,
        use_multithreading=True,
        max_concurrency=8
    )

    docs = loader.load()

    if not docs:
        print("⚠ No PDF documents found.")
        return {
            "status": "empty",
            "file_path": None,
            "total_pdfs": 0
        }

    print("\n--- Metadata Preview (First Page) ---")
    pprint(docs[0].metadata)
    print("-" * 40 + "\n")

    data = []
    for doc in docs:
        meta = doc.metadata

        data.append({
            "pdf_name": os.path.basename(meta.get("source", "unknown")),
            "path": meta.get("source"),
            "author": meta.get("author", "N/A"),
            "subject": meta.get("subject", "N/A"),
            "page_number": meta.get("page", 0) + 1,
            "content": doc.page_content
        })

    df = pd.DataFrame(data)

    df_combined = (
        df.groupby(['pdf_name', 'path'])
        .agg({
            'content': lambda x: " ".join(x),
            'author': 'first',
            'subject': 'first'
        })
        .reset_index()
    )

    df_combined.to_csv(output_filename, index=False)

    total = len(df_combined)

    print(f"--- Done! Processed {total} PDFs into '{output_filename}' ---")

    return {
        "status": "success",
        "file_path": output_filename,
        "total_pdfs": total
    }

