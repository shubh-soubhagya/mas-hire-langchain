# main.py

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

from pipeline_tools import PIPELINE_TOOLS


# ---------------------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------------------

load_dotenv()


# ---------------------------------------------------
# LLM (MASTER BRAIN)
# ---------------------------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------------------------------
# CREATE MASTER AGENT
# ---------------------------------------------------

agent = create_agent(
    model=llm,
    tools=PIPELINE_TOOLS
)


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------

if __name__ == "__main__":

    system_message = SystemMessage(
        content="""
You are an autonomous recruitment pipeline agent.

EXECUTION RULES:
1. You MUST execute the workflow sequentially.
2. DO NOT call multiple tools at once. Call ONE tool, wait for the result, then decide the next step.
3. If a tool output indicates success (e.g., path to a CSV), use that path for the NEXT tool.
4. Once 'send_emails_to_shortlisted' is complete, STOP and provide a final summary. DO NOT start the pipeline again.

STEPS:
1. Extract PDFs: process_cv_directory
2. Extract Details: extract_candidates
3. Match Jobs: match_candidates_to_jobs
4. Notify: send_emails_to_shortlisted
"""
    )

    human_message = HumanMessage(
        content="""
Start the recruitment pipeline:
- Source: ./CVDemo
- Output: outputs/fast_pdf_metadata.csv
- Job Descriptions: ./JD/job_description_demo.csv

Process the CVs, extract details, match them, and email the candidates.
"""
    )

    print("\n🚀 Starting Agentic Recruitment Pipeline...\n")

    result = agent.invoke({
        "messages": [
            system_message,
            human_message
        ]
    })

    print("\n✅ FINAL RESULT:\n")

    # Get last AI message (final summary)
    final_message = result["messages"][-1].content

    print(final_message)