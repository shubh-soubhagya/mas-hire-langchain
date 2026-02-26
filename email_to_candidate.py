import os
import base64
import pandas as pd
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText

# --- CONFIGURATION ---
# The file path to your Google Cloud JSON credentials
CLIENT_SECRET_FILE = r'credentials\credentials_email.json' 
SCOPES = ['https://www.googleapis.com/auth/gmail.send']
CSV_PATH = r"outputs\fast_pdf_metadata.csv"

def get_gmail_service():
    """Authenticates the user and returns the Gmail service object."""
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)

def send_message(service, user_id, message):
    """Sends the actual email message."""
    try:
        message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"--- Message Id: {message['id']} sent successfully! ---")
    except Exception as error:
        print(f"An error occurred: {error}")

def create_message(sender, to, subject, message_text):
    """Creates a MIME message for the Gmail API."""
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}

def draft_email_content(name, role):
    """Drafts a personalized email for the 2026 Batch candidates."""
    return f"""
Dear {name},

Greetings from the Recruitment Team at PrashnaAI!

We have reviewed your application and are impressed with your technical background and alignment with our requirements for the {role} position. We are pleased to officially shortlist you for the next round of interviews.

As part of our 2026 Batch hiring initiative, we are seeking candidates to contribute to our Agentic AI systems. Your profile stands out as a strong match for our context-aware engineering needs.

Please share your availability for a brief introductory call this week.

Best Regards,

Recruitment Team | PrashnaAI
    """

def run_email_pipeline():
    # Load data
    if not os.path.exists(CSV_PATH):
        print("CSV file not found.")
        return
        
    df = pd.read_csv(CSV_PATH)

    # Filter for Shortlisted Candidates (Score >= 80%)
    # We clean the percentage string to perform a numeric comparison
    df['score_numeric'] = df['match_score'].str.replace('%', '').astype(float)
    shortlisted = df[df['score_numeric'] >= 80].copy()

    if shortlisted.empty:
        print("No candidates found with a score of 80% or higher.")
        return

    print("\n--- Shortlisted Candidates (Score >= 80%) ---")
    for i, row in shortlisted.reset_index().iterrows():
        print(f"{i + 1}. {row['candidate_name']} (Score: {row['match_score']}) - Role: {row['best_matched_job']}")

    choice = int(input("\nEnter the number of the candidate you want to email (or 0 to exit): "))
    
    if choice == 0:
        return

    selected_candidate = shortlisted.iloc[choice - 1]
    
    # Initialize Gmail Service
    service = get_gmail_service()
    
    # Prepare Content
    email_body = draft_email_content(
        selected_candidate['candidate_name'], 
        selected_candidate['best_matched_job']
    )
    
    message = create_message(
        sender="me", # 'me' indicates the authenticated user
        to=selected_candidate['email_id'],
        subject=f"Shortlisted for {selected_candidate['best_matched_job']} | PrashnaAI",
        message_text=email_body
    )

    # Send
    print(f"Sending email to {selected_candidate['candidate_name']} ({selected_candidate['email_id']})...")
    send_message(service, "me", message)

if __name__ == "__main__":
    run_email_pipeline()