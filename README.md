# LexiVault

Legal document question answering with cited sources. Hybrid retrieval (FAISS + BM25 + reranker),
Tavily web search on trusted legal sites, and Gemini answers grounded in the evidence.
General information only, not legal advice.

Run: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`
(needs `GEMINI_API_KEY` and `TAVILY_API_KEY` in `.streamlit/secrets.toml`).
