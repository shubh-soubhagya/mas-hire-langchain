import os
from candidate_cv_extraction import fast_process_pdfs
from candidate_data_extraction import update_csv_with_info
from jd_matching_candidate_score import run_matching_pipeline
from email_to_candidate import run_email_pipeline

def main():
    # --- CONFIGURATION ---
    CV_DIRECTORY = "CVDemo"
    OUTPUT_CSV = os.path.join("outputs", "fast_pdf_metadata.csv")
    JD_FILE = os.path.join("JD", "job_description_Demo.csv")

    # Ensure output directory exists
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    print("\n" + "="*50)
    print("STEP 1: CV EXTRACTION")
    print("="*50)
    # 1. Extraction of text from PDFs
    fast_process_pdfs(CV_DIRECTORY, OUTPUT_CSV)

    print("\n" + "="*50)
    print("STEP 2: DATA EXTRACTION (Name, Email, Phone)")
    print("="*50)
    # 2. Extracting structured data using LLM
    update_csv_with_info(OUTPUT_CSV)

    print("\n" + "="*50)
    print("STEP 3: JD MATCHING & SCORING")
    print("="*50)
    # 3. Matching candidates against Job Descriptions
    run_matching_pipeline(OUTPUT_CSV, JD_FILE)

    print("\n" + "="*50)
    print("STEP 4: EMAIL SHORTLISTED CANDIDATES")
    print("="*50)
    # 4. Emailing shortlisted candidates (Interactive Selection)
    run_email_pipeline()

    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
