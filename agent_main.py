import os
import re
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tools_agentic import (
    extract_cv_text, 
    enrich_candidate_info, 
    match_candidates_to_jd, 
    list_shortlisted_candidates, 
    send_email_to_candidate, 
    get_pipeline_status
)

# Load environment variables
load_dotenv()

# Dictionary of tools for easy lookup
TOOL_MAP = {
    "get_pipeline_status": get_pipeline_status,
    "extract_cv_text": extract_cv_text,
    "enrich_candidate_info": enrich_candidate_info,
    "match_candidates_to_jd": match_candidates_to_jd,
    "list_shortlisted_candidates": list_shortlisted_candidates,
    "send_email_to_candidate": send_email_to_candidate
}

def run_recruitment_agent(query: str):
    # Initialize LLM
    llm = ChatOllama(model="gemma2:2b", temperature=0, stop=["Observation:"])
    
    system_prompt = """You are a Recruitment Agent. CALL TOOLS ONE BY ONE.
Available Tools:
- get_pipeline_status (No args)
- extract_cv_text (No args)
- enrich_candidate_info (No args)
- match_candidates_to_jd (No args)
- list_shortlisted_candidates (No args)
- send_email_to_candidate {"candidate_name": "...", "candidate_email": "...", "job_title": "..."}

STRICT FORMAT:
Action: [tool_name]
Action Input: [JSON arguments or {}]

1. Use 'get_pipeline_status' first.
2. If status says 0 CVs, call 'extract_cv_text'.
3. Use 'list_shortlisted_candidates' after matching.
4. Email all candidates one by one.
5. Provide a 'Final Answer:' once everything is done.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    print("\n--- Starting Recruitment Agent ---")
    
    for i in range(15):
        try:
            response = llm.invoke(messages)
            content = response.content
            print(f"\n[Step {i+1}] LLM Output:\n{content.strip()}")
            messages.append(response)
            
            if "Final Answer:" in content:
                print("\nTASK COMPLETED")
                break
                
            # Flexible regex to handle bolding like **Action:** or Action:
            action_match = re.search(r"Action:\s*\*?(\w+)\*?", content, re.IGNORECASE)
            args_match = re.search(r"Action Input:\s*\*?(\{.*\})\*?", content, re.IGNORECASE)
            
            if action_match:
                tool_name = action_match.group(1).strip()
                # Clean tool name in case LLM added extra markdown or chars
                tool_name = re.sub(r'[^a-zA-Z0-0_]', '', tool_name)
                
                inputs = {}
                if args_match:
                    try:
                        inputs = json.loads(args_match.group(1).strip())
                    except:
                        pass
                
                if tool_name in TOOL_MAP:
                    print(f"Calling tool: {tool_name}...")
                    if tool_name == "send_email_to_candidate":
                        obs = TOOL_MAP[tool_name](
                            inputs.get("candidate_name"), 
                            inputs.get("candidate_email"), 
                            inputs.get("job_title")
                        )
                    else:
                        obs = TOOL_MAP[tool_name]()
                    print(f"Observation: {obs}")
                    messages.append(HumanMessage(content=f"Observation: {obs}"))
                else:
                    messages.append(HumanMessage(content=f"Error: Tool '{tool_name}' not found."))
            else:
                messages.append(HumanMessage(content="Error: Please specify an Action or Final Answer."))
        except Exception as e:
            print(f"Loop error: {e}")
            messages.append(HumanMessage(content=f"Error processing your last response: {str(e)}"))

if __name__ == "__main__":
    user_query = "Process the hiring pipeline: extract, enrich, match, and email candidates >= 80%."
    run_recruitment_agent(user_query)
