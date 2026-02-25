import os
import pandas as pd
from pprint import pprint
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader


CANDIDATE_DATA = r"outputs\fast_pdf_metadata.csv"

def fast_process_pdfs(directory_path, output_filename = CANDIDATE_DATA):
    print(f"--- Starting Fast Extraction from: {directory_path} ---")
    
    # Initialize the high-speed loader
    loader = DirectoryLoader(
        directory_path, 
        glob="./*.pdf", 
        loader_cls=PyMuPDFLoader,
        use_multithreading=True,
        max_concurrency=8
    )

    # Load all documents
    docs = loader.load()

    if not docs:
        print("No PDF documents found.")
        return

    # --- PPRINT SECTION ---
    print("\n--- Metadata Preview for the First Page ---")
    # We pprint the metadata of the first document object found
    pprint(docs[0].metadata)
    print("-" * 40 + "\n")

    # Create list of dicts with expanded metadata
    data = []
    for doc in docs:
        meta = doc.metadata
        data.append({
            "pdf_name": os.path.basename(meta.get("source", "unknown")),
            "path": meta.get("source"),
            "author": meta.get("author", "N/A"),
            "subject": meta.get("subject", "N/A"),
            "page_number": meta.get("page", 0) + 1, # Page index usually starts at 0
            "content": doc.page_content
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Aggregate: Combine content AND keep the primary metadata
    # We take 'first' for metadata fields since they are usually identical across pages
    df_combined = df.groupby(['pdf_name', 'path']).agg({
        'content': lambda x: ' '.join(x),
        'author': 'first',
        'subject': 'first'
    }).reset_index()

    # Save to CSV
    df_combined.to_csv(output_filename, index=False)
    print(f"--- Done! Processed {len(df_combined)} PDFs into '{output_filename}' ---")

if __name__ == "__main__":
    # Using your specific directory 'CVDemo'
    fast_process_pdfs(r"CVDemo")
    