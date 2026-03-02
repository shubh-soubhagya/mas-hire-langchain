import os
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

load_dotenv()


# ---------------- SCHEMA ----------------

class MatchSchema(BaseModel):
    """Schema for job-resume matching results."""
    score: int = Field(description="Match score between 0 and 100")
    reason: str = Field(
        description="Two sentences: S1 shared skills. S2 main fit or gap."
    )
    job_title: str = Field(description="Job title evaluated")


# ---------------- TOOL ----------------

@tool(args_schema=MatchSchema)
def record_job_match(score: int, reason: str, job_title: str) -> str:
    """Registers a job match evaluation."""
    return f"Match Recorded: {job_title} - Score: {score}%"


# ---------------- AGENT (SIMPLIFIED TO LLM) ----------------

# ✅ Single LLM instance for matching
EVAL_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
).with_structured_output(MatchSchema)


# ---------------- EVALUATION ----------------

def evaluate_fit(resume_text: str, jd_text: str, job_title: str) -> dict:
    """
    Uses LLM with structured output to evaluate resume vs JD.
    """

    prompt = (
        f"You are a Precision Recruitment Auditor.\n"
        f"Evaluate resume against job role: {job_title}.\n\n"
        "Rules:\n"
        "1. No hallucinations\n"
        "2. Identify skill overlap and gaps\n"
        "3. Rank fit strictly between 0-100\n\n"
        f"JOB DESCRIPTION:\n{jd_text[:1500]}\n\n"
        f"RESUME:\n{resume_text[:2000]}"
    )

    try:
        result = EVAL_LLM.invoke(prompt)
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        return {
            "status": "error",
            "score": 0,
            "reason": str(e),
            "job_title": job_title
        }


# ---------------- PIPELINE ----------------

def run_matching_pipeline(file_path: str, jd_path: str):
    """Run matching between candidate CSV and JD CSV."""
    
    print(f"\n--- Loading CSVs for matching ---")
    
    # Robust CSV reading
    try:
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
    except Exception as e:
        print(f"Warning: Primary CSV read failed ({e}). Trying with latin1...")
        df = pd.read_csv(file_path, encoding='latin1')

    try:
        j_df = pd.read_csv(jd_path, encoding='utf-8', encoding_errors='replace')
    except Exception as e:
        print(f"Warning: JD CSV read failed ({e}). Trying with latin1...")
        j_df = pd.read_csv(jd_path, encoding='latin1')

    if df.empty:
        print("⚠ Candidate Dataframe is empty. Aborting matching.")
        return

    # Initialize columns if missing
    for col in ['match_score', 'match_reason', 'best_matched_job']:
        if col not in df.columns:
            df[col] = None

    print(f"--- Matching {len(df)} candidates against {len(j_df)} jobs ---")

    for idx, c_row in df.iterrows():
        print(f"Processing Candidate {idx+1}/{len(df)}...")
        
        best_score = -1
        best_match = {"score": "0%", "reason": "No match found", "title": "N/A"}

        # Optimization: Only process first 10 jobs to avoid long delays
        job_limit = min(len(j_df), 10)
        
        for j_idx, j_row in j_df.head(job_limit).iterrows():
            print(f"  Evaluating candidate {idx+1} against job {j_idx+1} ({j_row['Job Title']})...")
            
            # Basic sanity check for content
            if pd.isna(c_row.get('content')) or str(c_row['content']).strip() == "":
                continue
                
            result = evaluate_fit(
                resume_text=str(c_row['content']),
                jd_text=str(j_row.get('Job Description', '')),
                job_title=str(j_row.get('Job Title', 'Unknown'))
            )

            if result["status"] == "success":
                data = result["result"]
                current_score = data.score
                if current_score > best_score:
                    best_score = current_score
                    best_match = {
                        "score": f"{current_score}%",
                        "reason": data.reason,
                        "title": data.job_title
                    }

        df.at[idx, 'match_score'] = best_match.get("score")
        df.at[idx, 'match_reason'] = best_match.get("reason")
        df.at[idx, 'best_matched_job'] = best_match.get("title")

    # Final save
    df.to_csv(file_path, index=False)
    print(f"✅ Done! Matching results saved to {file_path}\n")
