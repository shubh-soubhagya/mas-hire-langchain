from langchain.tools import tool
from langgraph.types import interrupt


@tool
def request_human_email_approval(message: str) -> str:
    """
    Pause pipeline and request human approval
    before sending emails.
    """

    # Pause execution here
    response = interrupt({
        "type": "approval_required",
        "message": message,
        "action": "login_gmail_and_confirm"
    })

    # Execution resumes AFTER human response
    return response.get("status", "approved")