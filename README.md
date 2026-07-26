# PDF RAG Chatbot

A Streamlit chatbot backed by a LangGraph agent that can answer questions
about an uploaded PDF (via FAISS retrieval) or search the web, with
per-thread conversation history persisted in SQLite.

## Setup

1. Clone the repo and enter the folder.
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your own keys:
   ```
   HUGGINGFACEHUB_API_TOKEN=...   # read-only token from huggingface.co/settings/tokens
   GROQ_API_KEY=...               # from console.groq.com
   ```
4. Run the app:
   ```
   streamlit run ragF.py
   ```

## Notes

- Uploaded PDFs are chunked, embedded, and stored as a FAISS index under
  `faiss_indexes/<thread_id>/` (git-ignored, generated at runtime).
- Conversation history is stored in `chatbot.db` (git-ignored, generated at runtime).
- Never commit `.env` — it holds your real API keys.
