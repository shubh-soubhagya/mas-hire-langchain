import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()


class CandidateInfo(BaseModel):
    """Schema for candidate contact details extraction."""
    name: str = Field(description="Full name of the candidate")
    phone: str = Field(description="Full phone number exactly as written")
    email: str = Field(description="Email address of the candidate")

@tool(args_schema=CandidateInfo)
def extract_candidate_details_tool(name: str, phone: str, email: str) -> str:
    """Stores extracted candidate information."""
    return f"Processed candidate: {name} | {email} | {phone}"

EXTRACTOR_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
).with_structured_output(CandidateInfo)

def process_resume_content(content: str) -> dict:
    """
    Processes resume text using direct LLM call with structured output.
    """

    text_sample = content[:4000]

    prompt = (
        "You are a Precision Data Extraction specialist.\n"
        "Extract candidate Name, Phone, and Email from the resume.\n"
        "If you cannot find a field, use 'N/A'.\n\n"
        f"RESUME TEXT:\n{text_sample}"
    )

    try:
        result = EXTRACTOR_LLM.invoke(prompt)
        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
