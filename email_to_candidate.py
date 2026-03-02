import os
import base64
import pandas as pd

from email.mime.text import MIMEText
from dotenv import load_dotenv

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.tools import tool

load_dotenv()

CLIENT_SECRET_FILE = r'credentials/credentials_email.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
CSV_PATH = r"outputs/fast_pdf_metadata.csv"

def get_gmail_service():
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        SCOPES
    )
    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)


def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}


def send_message(service, user_id, message):
    message = service.users().messages().send(
        userId=user_id,
        body=message
    ).execute()

    return message['id']

def draft_email_content(name, role):
    return f"""
Dear {name},

Greetings from the Recruitment Team at PrashnaAI!

We have reviewed your application and are impressed with your background
for the {role} position.

You are shortlisted for the next round of interviews.

Please share your availability for a brief introductory call.

Best Regards,
Recruitment Team | PrashnaAI
"""

@tool
def send_shortlisted_emails(csv_path: str = CSV_PATH,
                            min_score: int = 80) -> str:
    """
    Sends emails automatically to candidates whose match score
    is above the given threshold.
    """

    if not os.path.exists(csv_path):
        return f"ERROR: CSV file not found at {csv_path}."

    print(f"\n--- Sending emails to shortlisted from {csv_path} ---")

    try:
        df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    except Exception:
        df = pd.read_csv(csv_path, encoding='latin1')

    if "match_score" not in df.columns:
        return f"ERROR: 'match_score' not found in {csv_path}. Run matching first."

    try:
        df['score_numeric'] = (
            df['match_score']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
        )

        df['score_numeric'] = pd.to_numeric(
            df['score_numeric'],
            errors="coerce"   
        )

    except Exception as e:
        return f"ERROR: Failed to parse match_score: {e}"

    shortlisted = df[df['score_numeric'] >= min_score]

    if shortlisted.empty:
        return "No shortlisted candidates (scores below threshold)."


    service = get_gmail_service()

    sent_count = 0

    for _, row in shortlisted.iterrows():

        email_body = draft_email_content(
            row['candidate_name'],
            row['best_matched_job']
        )

        message = create_message(
            sender="me",
            to=row['email_id'],
            subject=f"Shortlisted for {row['best_matched_job']} | PrashnaAI",
            message_text=email_body
        )

        send_message(service, "me", message)
        sent_count += 1

    return f"Emails successfully sent to {sent_count} shortlisted candidates."