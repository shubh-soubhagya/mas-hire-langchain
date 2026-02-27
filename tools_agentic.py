import os
import pandas as pd
from langchain_core.tools import tool
from candidate_cv_extraction import fast_process_pdfs
from candidate_data_extraction import update_csv_with_info
from jd_matching_candidate_score import run_matching_pipeline
from email_to_candidate import get_gmail_service, create_message, send_message, draft_email_content

# Configuration constants
CV_DIRECTORY = "CVDemo"
OUTPUT_CSV = os.path.join("outputs", "fast_pdf_metadata.csv")
JD_FILE = os.path.join("JD", "job_description_Demo.csv")

@tool
def extract_cv_text():
    """Extracts text content from all PDF resumes in the CVDemo directory and saves them to a CSV file."""
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    fast_process_pdfs(CV_DIRECTORY, OUTPUT_CSV)
    return f"Successfully extracted text from CVs in {CV_DIRECTORY} and saved to {OUTPUT_CSV}."

@tool
def enrich_candidate_info():
    """Uses an LLM to extract structured data (Name, Email, Phone) from the extracted CV text in the output CSV."""
    if not os.path.exists(OUTPUT_CSV):
        return "Error: extracted CV metadata CSV not found. Run extract_cv_text first."
    update_csv_with_info(OUTPUT_CSV)
    return f"Successfully enriched candidate info in {OUTPUT_CSV}."

@tool
def match_candidates_to_jd():
    """Matches the enriched candidate data against the Job Description (JD) and calculates compatibility scores."""
    if not os.path.exists(OUTPUT_CSV):
        return "Error: candidate data CSV not found. Run extract_cv_text and enrich_candidate_info first."
    if not os.path.exists(JD_FILE):
        return f"Error: Job Description file not found at {JD_FILE}."
    run_matching_pipeline(OUTPUT_CSV, JD_FILE)
    return f"Successfully matched candidates against JD and updated scores in {OUTPUT_CSV}."

@tool
def list_shortlisted_candidates():
    """Reads the candidate CSV and returns a list of candidates who have a match score of 80% or higher."""
    if not os.path.exists(OUTPUT_CSV):
        return "Error: candidate data CSV not found. Run the matching process first."
    
    df = pd.read_csv(OUTPUT_CSV)
    if 'match_score' not in df.columns:
        return "Error: match_score column missing. Run match_candidates_to_jd first."
    
    # Process scores reliably
    df['score_numeric'] = df['match_score'].astype(str).str.replace('%', '', regex=False)
    # Filter out non-numeric or empty scores if any
    df = df[df['score_numeric'].str.replace('.', '', count=1).str.isdigit()]
    df['score_numeric'] = df['score_numeric'].astype(float)
    
    shortlisted = df[df['score_numeric'] >= 80].copy()
    
    if shortlisted.empty:
        return "No candidates found with a score of 80% or higher."
    
    results = []
    for _, row in shortlisted.iterrows():
        results.append({
            "name": row['candidate_name'],
            "email": row['email_id'],
            "score": row['match_score'],
            "job": row.get('best_matched_job', 'N/A')
        })
    return results

@tool
def send_email_to_candidate(candidate_name: str, candidate_email: str, job_title: str):
    """Sends a shortlisting email to a specific candidate. 
    Requires candidate_name, candidate_email, and the job_title they were matched for."""
    try:
        service = get_gmail_service()
        email_body = draft_email_content(candidate_name, job_title)
        message = create_message(
            sender="me",
            to=candidate_email,
            subject=f"Shortlisted for {job_title} | PrashnaAI",
            message_text=email_body
        )
        send_message(service, "me", message)
        return f"Email successfully sent to {candidate_name} ({candidate_email}) for role {job_title}."
    except Exception as e:
        return f"Failed to send email to {candidate_name}: {str(e)}"

@tool
def get_pipeline_status():
    """Checks the overall status of the hiring pipeline by inspecting the output files."""
    status = {}
    
    # 1. Check extraction
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        status['cvs_extracted'] = len(df)
        
        # 2. Check enrichment
        if 'candidate_name' in df.columns:
            enriched_count = len(df[df['candidate_name'] != "Not Found"])
            status['candidates_enriched'] = enriched_count
        else:
            status['candidates_enriched'] = 0
            
        # 3. Check matching
        if 'match_score' in df.columns:
            matched_count = len(df[df['match_score'].notna()])
            status['candidates_matched'] = matched_count
            
            # 4. Check shortlisting
            df['score_numeric'] = df['match_score'].astype(str).str.replace('%', '', regex=False)
            df = df[df['score_numeric'].str.isdigit()] if not df.empty else df
            if not df.empty:
                df['score_numeric'] = df['score_numeric'].astype(float)
                shortlisted_count = len(df[df['score_numeric'] >= 80])
                status['shortlisted_count'] = shortlisted_count
        else:
            status['candidates_matched'] = 0
            status['shortlisted_count'] = 0
    else:
        status['cvs_extracted'] = 0
        status['candidates_enriched'] = 0
        status['candidates_matched'] = 0
        status['shortlisted_count'] = 0
        
    return (
        f"Pipeline Status:\n"
        f"- CVs Extracted: {status['cvs_extracted']}\n"
        f"- Candidates Enriched (Name/Email found): {status['candidates_enriched']}\n"
        f"- Candidates Matched with JD: {status['candidates_matched']}\n"
        f"- Candidates Shortlisted (>= 80%): {status['shortlisted_count']}"
    )
