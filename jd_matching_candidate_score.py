import pandas as pd
import time
import os
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tqdm import tqdm # Install via: pip install tqdm

class MatchSchema(BaseModel):
    score: int = Field(description="Match score 0-100")
    reason: str = Field(description="Short 2-sentence explanation")

# Setup Model
llm = ChatOllama(model="gemma2:2b", temperature=0)
parser = JsonOutputParser(pydantic_object=MatchSchema)

@tool
def calculate_job_fit(resume_text: str, jd_text: str) -> Dict:
    """Compares a resume against a JD with internal retry logic."""
    
    # ADD THE ADAPTABLE PROMPT HERE
    system_prompt = (
        "You are a Precision Recruitment Auditor. Your task is to match a candidate to a specific Job Description (JD).\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. ADAPTABILITY: Do not assume every role is a Software Engineer. Use the specific Job Title from the JD to evaluate fit.\n"
        "2. NO HALLUCINATIONS: Only mention skills, tools, or experience explicitly stated in the Resume. "
        "If a skill is in the JD but NOT in the Resume, treat it as a gap.\n"
        "3. REASONING: Provide a 2-sentence explanation. "
        "Sentence 1 must list specific technical keywords shared by both. "
        "Sentence 2 must state the primary reason for the score based on the specific JD requirements.\n"
        "4. BATCH YEAR: Prioritize Skills and Experience.\n\n"
        "{format_instructions}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "TARGET JOB DESCRIPTION:\n{jd}\n\nCANDIDATE RESUME CONTENT:\n{resume}")
    ])
    
    chain = prompt | llm | parser
    
    for attempt in range(2):
        try:
            return chain.invoke({
                "resume": resume_text[:2000],
                "jd": jd_text[:1500],
                "format_instructions": parser.get_format_instructions()
            })
        except Exception:
            time.sleep(0.5)
    return {"score": 0, "reason": "Failed to extract"}


def match_single_candidate(args):
    c_idx, c_row, j_df = args
    best_score = -1
    best_data = {}

    for _, j_row in j_df.iterrows():
        res = calculate_job_fit.func(str(c_row['content']), str(j_row['Job Description']))
        
        if res['score'] > best_score:
            best_score = res['score']
            best_data = {
                "match_score": f"{res['score']}%",
                "match_reason": res['reason'],
                "best_matched_job": j_row['Job Title']
            }
            
    return best_data

def run_matching_pipeline(file_path, jd_path):
    def safe_load(path):
        try:
            return pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='cp1252')

    df = safe_load(file_path)
    j_df = safe_load(jd_path)
    
    # Ensure necessary columns exist
    for col in ['match_score', 'match_reason', 'best_matched_job']:
        if col not in df.columns:
            df[col] = None

    # CHECKING MECHANISM: Identify rows where data is missing
    # We filter for rows where 'match_score' is null or empty
    mask = df['match_score'].isna() | (df['match_score'] == "")
    rows_to_process = df[mask]

    if rows_to_process.empty:
        print("--- All rows already processed. No new data to find. ---")
        return

    print(f"--- Processing {len(rows_to_process)} remaining candidates ---")

    tasks = [(idx, row, j_df) for idx, row in rows_to_process.iterrows()]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        list_results = list(tqdm(
            executor.map(match_single_candidate, tasks), 
            total=len(rows_to_process), 
            desc="Updating Rows"
        ))

    # Update only the specific rows in the original dataframe
    for i, (idx, _, _) in enumerate(tasks):
        df.at[idx, 'match_score'] = list_results[i]['match_score']
        df.at[idx, 'match_reason'] = list_results[i]['match_reason']
        df.at[idx, 'best_matched_job'] = list_results[i]['best_matched_job']

    # Save directly back to the source file
    df.to_csv(file_path, index=False)
    print(f"\n--- Success! {file_path} updated in-place. ---")

if __name__ == "__main__":
    # Now using the same file as both input and output
    target_file = r"outputs\fast_pdf_metadata.csv"
    jd_file = r"JD\job_description_Demo.csv"
    run_matching_pipeline(target_file, jd_file)