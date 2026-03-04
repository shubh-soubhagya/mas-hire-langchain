# mas-hire-langchain

## Overview
A fully autonomous recruitment pipeline built with **LangChain**, **Groq** LLMs, and **pandas**. The system extracts candidate information from PDF resumes, matches them against job descriptions, and automatically emails shortlisted candidates.

## Features
- Fast PDF metadata extraction
- Structured LLM extraction of candidate name, email, and phone
- Job‑description matching with score, reason, and best‑matched job title
- Automated email notifications via Gmail API
- Sequential, deterministic workflow to avoid infinite loops

## Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/shubh-soubhagya/mas-hire-langchain.git
   cd mas-hire-langchain
   ```
2. **Create a virtual environment & install dependencies**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   - Copy the provided `.env.example` to `.env` (or create one manually).
   - Add your `GROQ_API_KEY`.
   - Add your Gmail `CLIENT_ID`, `CLIENT_SECRET`, and `REFRESH_TOKEN` from the Google Cloud Console.

## Google Cloud API Credentials
 To use the automated email feature, you must set up a Google Cloud Project with the Gmail API enabled:

1.  **Create a Project**: Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  **Enable Gmail API**: Search for "Gmail API" and click **Enable**.
3.  **Configure OAuth Consent Screen**:
    - Select "External" (or "Internal" if using a workspace).
    - Add `https://www.googleapis.com/auth/gmail.send` to scopes.
    - Add your email as a **Test User**.
4.  **Create Credentials**:
    - Go to **Credentials** > **Create Credentials** > **OAuth client ID**.
    - Select **Desktop app**.
    - Download the `json` file or copy the `Client ID` and `Client Secret`.
5.  **Get Refresh Token**: Use a script or the [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/) to generate a long-lived `REFRESH_TOKEN` with the `https://www.googleapis.com/auth/gmail.send` scope.

## Usage
Run the pipeline with a single command:
```bash
python .\main.py
```
The agent will:
1. Extract PDFs from `./CVDemo` → `outputs/fast_pdf_metadata.csv`
2. Extract candidate details (name, email, phone) and add them to the CSV
3. Match each candidate against the job descriptions in `./JD/job_description_demo.csv`
4. Email candidates whose match score meets the threshold (default 80%)

## Pipeline Steps (Sequential)
| Step | Tool | Description |
|------|------|-------------|
| 1 | `process_cv_directory` | Reads all PDFs and creates a CSV with raw content |
| 2 | `extract_candidates` | Calls the LLM to pull name, email, phone and updates the CSV |
| 3 | `match_candidates_to_jobs` | Scores each candidate against each JD, adds `match_score`, `match_reason`, `best_matched_job` |
| 4 | `send_emails_to_shortlisted` | Sends Gmail messages to candidates meeting the score threshold |

## Configuration
- **Model** – Change the LLM model in `main.py`, `candidate_data_extraction.py`, and `jd_matching_candidate_score.py` (default: `llama-3.3-70b-versatile`).
- **Threshold** – Adjust `min_score` in `email_to_candidate.py`.
- **CSV Paths** – Override source directories by editing the prompts in `main.py`.

## Contributing
1. Fork the repo
2. Create a feature branch
3. Ensure all tests pass (`pytest` if added)
4. Submit a pull request

## License
MIT License – see `LICENSE` for details.
