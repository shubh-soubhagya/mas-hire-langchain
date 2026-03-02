from langchain.tools import tool
import pandas as pd

from candidate_cv_extraction import fast_process_pdfs
from candidate_data_extraction import process_resume_content
from jd_matching_candidate_score import run_matching_pipeline
from email_to_candidate import send_shortlisted_emails



@tool
def process_cv_directory(directory_path: str) -> str:
    """Extract all CV PDFs from a directory and create a combined CSV file.
    Returns the path to the generated CSV file.
    """
    result = fast_process_pdfs(directory_path)
    if result["status"] != "success":
        return "ERROR: No PDFs processed"
    return result["file_path"]

@tool
def extract_candidates(csv_path: str) -> str:
    """Extract candidate details (name, email) from resume content in the CSV.
    Updates the CSV file with 'candidate_name' and 'email_id' columns.
    """
    # Agent may send dict or string
    if isinstance(csv_path, dict):
        csv_path = csv_path.get("csv_path")

    import os
    if not os.path.exists(csv_path):
        return f"ERROR: CSV not found at {csv_path}"

    print(f"\n--- Extracting Candidate Details from {csv_path} ---")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    except Exception:
        df = pd.read_csv(csv_path, encoding='latin1')

    if "content" not in df.columns:
        return "ERROR: 'content' column missing in CSV"

    # Initialize columns
    for col in ['candidate_name', 'email_id', 'phone_number']:
        if col not in df.columns:
            df[col] = "N/A"

    for idx, row in df.iterrows():
        print(f"Processing candidate {idx+1}/{len(df)}...")
        
        content = str(row.get("content", ""))
        if not content.strip():
            continue
            
        res = process_resume_content(content)
        
        if res["status"] == "success":
            data = res["result"]
            # Handle both dict and object return types
            name = data.get("name") if isinstance(data, dict) else getattr(data, "name", "N/A")
            email = data.get("email") if isinstance(data, dict) else getattr(data, "email", "N/A")
            phone = data.get("phone") if isinstance(data, dict) else getattr(data, "phone", "N/A")
            
            df.at[idx, 'candidate_name'] = name
            df.at[idx, 'email_id'] = email
            df.at[idx, 'phone_number'] = phone
            print(f"  Extracted: {name} ({email})")
        else:
            print(f"  Extraction failed: {res.get('message')}")

    # Save back to CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Extraction complete. Updated {csv_path}\n")
    
    return csv_path

@tool
def match_candidates_to_jobs(candidate_csv, jd_csv: str = "./JD/job_description_demo.csv") -> str:
    """Match candidates against job descriptions and compute scores.
    Returns the path to the updated candidate CSV file.
    If jd_csv is not provided, defaults to './JD/job_description_demo.csv'.
    """
    if isinstance(candidate_csv, dict):
        candidate_csv = candidate_csv.get("csv_path")
    # Ensure JD path exists, fallback to default if None or empty
    if not jd_csv:
        jd_csv = "./JD/job_description_demo.csv"
    run_matching_pipeline(candidate_csv, jd_csv)
    return candidate_csv


@tool
def send_emails_to_shortlisted(csv_path: str) -> str:
    """Send emails to shortlisted candidates based on match score.
    Returns a confirmation string.
    """

    # Invoke the StructuredTool with proper argument dict
    result = send_shortlisted_emails.invoke({"csv_path": csv_path})
    return result


PIPELINE_TOOLS = [
    process_cv_directory,
    extract_candidates,
    match_candidates_to_jobs,
    send_emails_to_shortlisted,
]