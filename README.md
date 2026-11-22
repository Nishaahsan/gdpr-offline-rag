# GDPR Assistant - Phase 1: Offline RAG

## Step 1 - Install Ollama (internet needed once)
Install from ollama.com, then:

    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

## Step 2 - Python environment

    python -m venv venv
    venv\Scripts\activate          (Linux/Mac: source venv/bin/activate)
    pip install -r requirements.txt

## Step 3 - Get the GDPR text
EUR-Lex (official): https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng
Select all text on the page, copy it, paste it into docs/gdpr.txt and SAVE (Ctrl+S).
Format check: each article line must be just "Article 17", and the next line its title.
Delete docs/PUT_GDPR_FILE_HERE.txt (placeholder).

## Step 4 - Build the index

    python ingest.py

You should see "99 articles found". If fewer, the missing article numbers are listed.

## Step 5 - Run the chatbot

    streamlit run app.py

## Test questions
- What is the deadline to notify a personal data breach?
- When can a data subject ask for erasure?
- What are the conditions for valid consent?
- What are the maximum fines?
- What is the capital of France?   (should answer: not covered)

## Troubleshooting
- **"SKIPPED ... only N characters":** the file is empty or a placeholder. Paste the GDPR text and save with Ctrl+S.
- **"articles missing" warning:** "Article 17" must be on its own line. Split "Article 17 Right to erasure" into two lines if they are joined.
- **0 articles found:** the file has no extractable text (scanned PDF). Copy-paste the EUR-Lex HTML page instead of using a PDF.
- **"Index not found":** run `python ingest.py` first.
- **Ollama connection error:** the Ollama app must be running (check with `ollama list`).
- **Test the chunker alone:** `python chunker.py` (no libraries needed).
