import pandas as pd
import time
from typing import Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# 1. Schema for extraction
class CandidateInfo(BaseModel):
    name: str = Field(description="Full name of the candidate")
    phone: str = Field(description="Full phone number including code and hyphens")
    email: str = Field(description="Email address")

# 2. Setup ChatOllama and Parser
# ChatOllama is generally more robust for structured output than OllamaLLM
llm = ChatOllama(model="gemma2:2b", temperature=0)
parser = JsonOutputParser(pydantic_object=CandidateInfo)

# 3. Enhanced Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data extractor. IMPORTANT: Return the FULL phone number exactly as written. It can be like +1-465-3587 or +918967010103. "
               "Do not shorten it. Wrap the phone number in double quotes. {format_instructions}"),
    ("human", "Extract from this text:\n\n{text}")
])

def extract_details(content, retries=10):
    """Feeds content to Gemma2 with automatic retry logic on failure."""
    attempt = 0
    text_sample = content[:2500] # Focus on the header where contact info lives
    
    while attempt < retries:
        try:
            chain = prompt | llm | parser
            result = chain.invoke({
                "text": text_sample,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Check if we actually got data or just placeholders
            name = result.get("name", "Not Found")
            phone = result.get("phone", "Not Found")
            email = result.get("email", "Not Found")
            
            # If at least one major piece of info is found, return it
            if name != "Not Found" or email != "Not Found":
                return name, phone, email
            
        except Exception as e:
            print(f"      Attempt {attempt + 1} failed. Retrying...")
            time.sleep(1) # Small breather for the local LLM
            
        attempt += 1
    
    return "Not Found", "Not Found", "Not Found"

def update_csv_with_info(file_path):
    print(f"--- Reading {file_path} ---")
    df = pd.read_csv(file_path)

    for col in ['candidate_name', 'phone_number', 'email_id']:
        if col not in df.columns:
            df[col] = "Not Found"

    print(f"--- Extracting info using Gemma2:2b (with Retries) ---")
    
    for index, row in df.iterrows():
        # Skip if already processed in a previous run
        if pd.notna(row['candidate_name']) and row['candidate_name'] != "Not Found":
            continue
            
        print(f"Processing row {index + 1}/{len(df)}: {row['pdf_name']}")
        
        name, phone, email = extract_details(row['content'])
        
        df.at[index, 'candidate_name'] = name
        df.at[index, 'phone_number'] = phone
        df.at[index, 'email_id'] = email
        
        # Save every 5 rows as a checkpoint
        if index % 5 == 0:
            df.to_csv(file_path, index=False)

    df.to_csv(file_path, index=False)
    print(f"--- Success! CSV updated at: {file_path} ---")

if __name__ == "__main__":
    csv_file = r"outputs\fast_pdf_metadata.csv" 
    update_csv_with_info(csv_file)