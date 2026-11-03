"""LexiVault (Streamlit): upload PDFs and ask questions. Hybrid RAG + Tavily + Gemini."""
import re, json, time, hashlib

import numpy as np
import faiss
import pymupdf  # PyMuPDF
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from google import genai
from google.genai import types

st.set_page_config(page_title="LexiVault", page_icon="⚖️")

# ---------------------------------------------------------------- settings
MODEL = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = st.secrets.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")

LIBRARY_THRESHOLD = 2.0   # built-in library, calibrated for the small reranker (raw scores)
UPLOAD_THRESHOLD = 0.0    # uploaded files: more lenient, since the user says the question is about their file
WEB_THRESHOLD = 0.3
TRUSTED_DOMAINS = ["indiankanoon.org", "indiacode.nic.in", "sci.gov.in"]
MAX_FILES, MAX_MB, MAX_PAGES, MAX_CHUNKS = 3, 10, 80, 500   # upload limits (free-tier memory and speed)
SESSION_LIMIT = 20            # questions per browser session
GLOBAL_LIMIT_PER_HOUR = 150   # questions per hour for the whole app

NO_EVIDENCE_MSG = ("I couldn't find reliable information on that in the documents or trusted legal "
                   "sources. Try a more specific question, for example about a particular clause, "
                   "section or term.")


def tokenize(text):
    return re.findall(r"\w+", text.lower()) or ["_"]


# ---------------------------------------------------------------- load models and the built-in library once
def make_store(chunks, faiss_index, threshold, cache):
    """A 'store' is everything needed to search one set of documents."""
    return {"chunks": chunks, "faiss": faiss_index,
            "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]),
            "threshold": threshold, "cache": cache}


@st.cache_resource(show_spinner="Loading models (first start takes a minute)...")
def load_resources():
    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return {
        "library": make_store(chunks, faiss.read_index("faiss_small.index"), LIBRARY_THRESHOLD, {}),
        "embed": SentenceTransformer("BAAI/bge-small-en-v1.5"),
        "rerank": CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "gemini": genai.Client(api_key=st.secrets["GEMINI_API_KEY"]),
        "tavily": TavilyClient(api_key=st.secrets["TAVILY_API_KEY"]),
        "web_cache": {}, "hits": [],
    }


R = load_resources()


# ---------------------------------------------------------------- PDF ingestion (for uploaded files)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def clean_text(text):
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"^\s*(Page\s*)?\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_heading(line):
    line = line.strip()
    if not line or len(line) > 80:
        return False
    numbered = re.match(r"^(\d+(\.\d+)*[.)]?|Section|Clause|Article)\s+\S+", line, re.IGNORECASE)
    return line.isupper() or bool(numbered)


def process_pdf(name, data):
    """PDF bytes -> (chunks, embeddings). Raises ValueError with a friendly message."""
    doc = pymupdf.open(stream=data, filetype="pdf")
    if len(doc) > MAX_PAGES:
        raise ValueError(f"{name} has {len(doc)} pages. The limit is {MAX_PAGES}.")
    chunks, heading = [], "Unknown"
    for page_number, page in enumerate(doc, start=1):
        cleaned = clean_text(page.get_text())
        if not cleaned:
            continue
        for piece in splitter.split_text(cleaned):
            for line in piece.split("\n"):
                if looks_like_heading(line):
                    heading = line.strip()
                    break
            chunks.append({"source": name, "page": page_number, "section": heading, "text": piece})
    if not chunks:
        raise ValueError(f"No readable text found in {name}. It may be a scanned PDF (OCR is not supported).")
    vecs = R["embed"].encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True,
                             show_progress_bar=False).astype("float32")
    return chunks, vecs


def update_user_store(uploaded_files):
    """Rebuild this session's searchable store whenever the set of uploaded files changes."""
    keys = {f"{f.name}|{hashlib.sha1(f.getvalue()).hexdigest()}": f for f in uploaded_files}
    sig = tuple(sorted(keys))
    if sig == st.session_state.sig:
        return
    previous = st.session_state.sig
    st.session_state.sig = sig

    errors, parts, names = [], [], []
    for i, (key, f) in enumerate(keys.items()):
        if i >= MAX_FILES:
            errors.append(f"Only the first {MAX_FILES} files are used. Remove some to use {f.name}.")
            break
        if f.size > MAX_MB * 1_000_000:
            errors.append(f"{f.name} is larger than {MAX_MB} MB.")
            continue
        if key not in st.session_state.file_cache:
            try:
                with st.spinner(f"Reading {f.name} (this takes a little while)..."):
                    st.session_state.file_cache[key] = process_pdf(f.name, f.getvalue())
            except ValueError as e:
                errors.append(str(e))
                continue
            except Exception:
                errors.append(f"Could not read {f.name}. Is it a valid PDF?")
                continue
        parts.append(st.session_state.file_cache[key])
        names.append(f.name)

    if sum(len(c) for c, _ in parts) > MAX_CHUNKS:
        errors.append("The files contain too much text for this free demo. Remove a file or use a shorter one.")
        parts, names = [], []

    st.session_state.file_cache = {k: v for k, v in st.session_state.file_cache.items() if k in keys}
    st.session_state.upload_errors = errors

    if parts:
        all_chunks = [c for chunks, _ in parts for c in chunks]
        vecs = np.vstack([v for _, v in parts])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        st.session_state.user_store = make_store(all_chunks, index, UPLOAD_THRESHOLD, {})
        st.session_state.user_store["names"] = names
        note = f"Now answering from your uploaded file(s): {', '.join(names)}."
    else:
        st.session_state.user_store = None
        note = "Back to the built-in document library."
    if parts or previous:
        st.session_state.messages.append({"role": "assistant", "content": note, "sources": []})


# ---------------------------------------------------------------- document search
def hybrid_search(store, question, top_k=20):
    top_k = min(top_k, len(store["chunks"]))
    q_vec = R["embed"].encode(
        ["Represent this sentence for searching relevant passages: " + question],
        normalize_embeddings=True).astype("float32")
    _, faiss_ids = store["faiss"].search(q_vec, top_k)
    bm25_ids = np.argsort(store["bm25"].get_scores(tokenize(question)))[::-1][:top_k]

    fused = {}
    for ranked_list in (faiss_ids[0], bm25_ids):
        for rank, idx in enumerate(ranked_list):
            fused[int(idx)] = fused.get(int(idx), 0) + 1 / (60 + rank)
    return sorted(fused, key=fused.get, reverse=True)[:top_k]


def search_documents(store, question, final_k=4):
    chunks = store["chunks"]
    candidates = hybrid_search(store, question)
    scores = R["rerank"].predict([(question, chunks[i]["text"]) for i in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:final_k]
    results = []
    for idx, score in ranked:
        item = dict(chunks[idx])
        item["score"] = float(score)
        results.append(item)
    return results


# ---------------------------------------------------------------- web search (cached)
def search_web(question, max_results=4):
    cache = R["web_cache"]
    if question in cache:
        return cache[question]
    results = []
    try:
        response = R["tavily"].search(question, max_results=max_results, include_domains=TRUSTED_DOMAINS)
        trusted = True
        if not response["results"]:
            response = R["tavily"].search(question, max_results=max_results)
            trusted = False
        for r in response["results"]:
            results.append({"title": r["title"], "url": r["url"], "text": r["content"],
                            "date": r.get("published_date"), "score": r.get("score", 0),
                            "trusted": trusted})
    except Exception as e:
        print("Web search failed:", e)
        return []
    cache[question] = results
    return results


# ---------------------------------------------------------------- routing and evidence
COMPARE_WORDS = ["compare", "comply", "compliance", "consistent", "current law", "legal under",
                 "valid under", "as per law", "against the law", "latest", "amendment", "violate"]

LEGAL_WORDS = ["law", "legal", "act", "section", "ipc", "bns", "crpc", "constitution", "article",
               "court", "judge", "judgment", "case", "bail", "fir", "arrest", "police", "offence",
               "offense", "crime", "penalty", "punishment", "fine", "contract", "agreement", "clause",
               "tender", "bid", "tenant", "landlord", "lease", "rent", "evict", "property", "mortgage",
               "deed", "gift", "will", "inheritance", "marriage", "divorce", "custody", "maintenance",
               "tax", "gst", "rights", "liability", "sue", "complaint", "notice", "tribunal",
               "consumer", "fraud", "cheating", "defamation", "registration", "stamp", "emd", "deposit"]


def route_question(question, has_doc_evidence):
    q = question.lower()
    if any(w in q for w in COMPARE_WORDS):
        return "both"
    return "documents" if has_doc_evidence else "web"


def is_legal_question(question):
    return any(w in set(re.findall(r"[a-z]+", question.lower())) for w in LEGAL_WORDS)


def gather_evidence(store, question):
    doc_results = [r for r in search_documents(store, question) if r["score"] >= store["threshold"]]
    route = route_question(question, has_doc_evidence=bool(doc_results))
    web_results = []
    if route in ("web", "both") and (doc_results or is_legal_question(question)):
        web_results = [w for w in search_web(question) if w["score"] >= WEB_THRESHOLD]
    return {"route": route, "doc_results": doc_results, "web_results": web_results,
            "enough_evidence": bool(doc_results or web_results)}


# ---------------------------------------------------------------- Gemini answer
def build_evidence_block(ev):
    lines, sources = [], {}
    for i, r in enumerate(ev["doc_results"], start=1):
        sid = f"D{i}"
        sources[sid] = {"type": "document", **r}
        lines.append(f"[{sid}] Document: {r['source']} | Page {r['page']} | Section: {r['section']}\n{r['text']}")
    for i, w in enumerate(ev["web_results"], start=1):
        sid = f"W{i}"
        sources[sid] = {"type": "web", **w}
        lines.append(f"[{sid}] Web: {w['title']} ({w['url']})\n{w['text'][:1200]}")
    return "\n\n".join(lines), sources


def generate_answer(question, evidence_text):
    prompt = f"""You are LexiVault, a legal document assistant.
Answer the question using ONLY the evidence below. Do not use outside knowledge.

Rules:
- Cite sources inline with their IDs, like [D1] or [W2].
- Document sources start with D, web sources start with W.
- If you use both, clearly separate "From your documents" and "From external legal sources".
- If the evidence does not answer the question, set "evidence_sufficient" to false and say so. Never invent clauses, legal provisions or citations.
- Keep the answer clear and concise. This is general information, not legal advice.

EVIDENCE:
{evidence_text}

QUESTION: {question}

Reply as JSON:
{{"answer": "...", "sources_used": ["D1", "W1"], "evidence_sufficient": true or false}}"""

    last_error = None
    for model in [MODEL, FALLBACK_MODEL]:
        for wait in [0, 3]:
            time.sleep(wait)
            try:
                response = R["gemini"].models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                return json.loads(response.text)
            except Exception as e:
                last_error = e
                if "503" not in str(e) and "UNAVAILABLE" not in str(e):
                    raise
                print(f"{model} is busy, retrying...")
    raise last_error


def validate(result, sources):
    answer = str(result.get("answer", "")).strip()
    declared = [s for s in result.get("sources_used", []) if s in sources]
    cited = [s for s in re.findall(r"\[([DW]\d+)\]", answer) if s in sources]
    used = sorted(set(declared + cited))
    if not result.get("evidence_sufficient", False):
        return (answer or NO_EVIDENCE_MSG), []
    if not used:
        return NO_EVIDENCE_MSG, []
    return answer, used


def to_source_line(sid, s):
    if s["type"] == "document":
        return {"text": f"{sid} · {s['source']}, page {s['page']}"}
    title = re.sub(r"[\[\]]", "", s["title"])
    if not s["trusted"]:
        title += " (not a trusted legal site, please verify)"
    url = s["url"] if s["url"].startswith(("http://", "https://")) else None
    return {"text": f"{sid} · {title}", "url": url}


def ask(store, question):
    cache = store["cache"]
    if question in cache:
        return cache[question]
    ev = gather_evidence(store, question)
    if not ev["enough_evidence"]:
        return {"answer": NO_EVIDENCE_MSG, "sources": [], "route": ev["route"]}

    evidence_text, sources = build_evidence_block(ev)
    try:
        result = generate_answer(question, evidence_text)
    except Exception as e:
        print("Gemini error:", e)
        return {"answer": "The AI model is busy or unavailable right now. Please wait a minute and ask again.",
                "sources": [], "route": ev["route"]}          # errors are not cached

    answer, used = validate(result, sources)
    output = {"answer": answer, "route": ev["route"],
              "sources": [to_source_line(sid, sources[sid]) for sid in used]}
    cache[question] = output
    return output


# ---------------------------------------------------------------- small talk (0 API calls)
def small_talk(text):
    t = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    if re.fullmatch(r"((hi+|hello+|hey+)\s+)?how are you( doing)?", t):
        return "I'm doing well, thank you! Upload a PDF or ask me about Indian law."
    if re.fullmatch(r"(good\s*)?morning( there)?", t):
        return "Good morning! 👋 Upload a PDF and ask me about it, or ask about Indian law."
    if re.fullmatch(r"(good\s*)?afternoon( there)?", t):
        return "Good afternoon! 👋 What would you like to know?"
    if re.fullmatch(r"(good\s*)?evening( there)?", t):
        return "Good evening! 👋 What would you like to know?"
    if re.fullmatch(r"(good\s*)?night", t):
        return "Good night! Come back anytime you have a legal question."
    if re.fullmatch(r"(h+i+|h+e+l+l*o+|h+e+y+|namaste|hola)( there| lexivault)?", t):
        return "Hello! 👋 I'm LexiVault. Upload a PDF in the sidebar and ask me about it, or ask about Indian law."
    if re.search(r"\b(thanks|thank you|thx)\b", t) and len(t.split()) <= 5:
        return "You're welcome! Let me know if you have another question."
    if re.fullmatch(r"(bye|goodbye|see you|see ya)( later)?", t):
        return "Goodbye! Take care. 👋"
    if re.search(r"who are you|what can you do|^help$", t):
        return ("I'm LexiVault, a legal document assistant. Upload a PDF and I'll answer questions from it, "
                "with page references. I can also look up general Indian law on trusted legal websites. "
                "If I can't find reliable evidence, I'll tell you.")
    return None


# ---------------------------------------------------------------- limits
def limit_reached():
    now = time.time()
    R["hits"][:] = [t for t in R["hits"] if now - t < 3600]
    if len(R["hits"]) >= GLOBAL_LIMIT_PER_HOUR:
        return "This demo has reached its hourly question limit. Please try again later."
    if st.session_state.asked >= SESSION_LIMIT:
        return "You've reached the question limit for this session. Refresh the page to continue."
    return None


# ---------------------------------------------------------------- UI
def show_message(msg):
    st.markdown(msg["content"].replace("$", "\\$"))
    if msg.get("sources"):
        st.caption("Supporting references")
        for s in msg["sources"]:
            st.markdown(f"- [{s['text']}]({s['url']})" if s.get("url") else f"- {s['text']}")


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.asked = 0
    st.session_state.sig = ()
    st.session_state.file_cache = {}
    st.session_state.user_store = None
    st.session_state.upload_errors = []

st.title("⚖️ LexiVault")
st.caption("Upload a legal PDF and ask questions about it, with page references. "
           "General information only, not legal advice.")

with st.sidebar:
    st.header("Your documents")
    uploaded = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    update_user_store(uploaded or [])
    for err in st.session_state.upload_errors:
        st.warning(err)
    st.caption(f"Up to {MAX_FILES} PDFs, {MAX_MB} MB and {MAX_PAGES} pages each. Text PDFs only (no scans). "
               "Files are processed in memory and not saved, but the relevant excerpts and your questions "
               "are sent to Google's Gemini API (and questions to Tavily for web search). "
               "Don't upload confidential documents.")

    user_store = st.session_state.user_store
    if user_store:
        st.success("Answering from: " + ", ".join(user_store["names"]))
        examples = ["What are the payment terms?", "What is the termination clause?",
                    "What are the obligations of each party?", "What is the penalty for breach?"]
    else:
        st.info("No file uploaded. Answering from the built-in library "
                "(a tender, a property guide, the Telangana rent Act, the Hindu Marriage Act).")
        examples = ["What is the earnest money deposit?", "What is a mortgage deed?",
                    "What are the conditions for a Hindu marriage?",
                    "Can a landlord evict a tenant under the rent control act?",
                    "What is Section 420 of the IPC?",
                    "Is the EMD in my tender consistent with general financial rules in India?"]

    st.divider()
    st.header("Try asking")
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    st.caption("Specific questions work best. Broad requests like \"summarize this\" don't suit this kind of search.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

store = st.session_state.user_store or R["library"]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        show_message(m)

question = st.chat_input("Ask a question about your document...") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question.replace("$", "\\$"))

    with st.chat_message("assistant"):
        reply = small_talk(question)
        if reply:
            msg = {"role": "assistant", "content": reply, "sources": []}
        elif (blocked := limit_reached()):
            msg = {"role": "assistant", "content": blocked, "sources": []}
        else:
            with st.spinner("Searching the documents..."):
                out = ask(store, question.strip())
            R["hits"].append(time.time())
            st.session_state.asked += 1
            msg = {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
        show_message(msg)
    st.session_state.messages.append(msg)
