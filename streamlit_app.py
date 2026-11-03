"""LexiVault (Streamlit version): hybrid RAG + Tavily + Gemini, built to fit free-tier memory."""
import re, json, time

import numpy as np
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tavily import TavilyClient
from google import genai
from google.genai import types

st.set_page_config(page_title="LexiVault", page_icon="⚖️")

# ---------------------------------------------------------------- settings
MODEL = st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = st.secrets.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")

RELEVANCE_THRESHOLD = 2.0     # calibrated for the small reranker (raw scores)
WEB_THRESHOLD = 0.3
TRUSTED_DOMAINS = ["indiankanoon.org", "indiacode.nic.in", "sci.gov.in"]
SESSION_LIMIT = 20            # questions per browser session
GLOBAL_LIMIT_PER_HOUR = 150   # questions per hour for the whole app

NO_EVIDENCE_MSG = ("I couldn't find reliable information on that in the documents or trusted legal "
                   "sources. I'm best at questions about the document library and Indian law. "
                   "Try rephrasing with specific legal terms, such as a section, clause or act name.")


def tokenize(text):
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------- load everything once
@st.cache_resource(show_spinner="Loading models (first start takes a minute)...")
def load_resources():
    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return {
        "chunks": chunks,
        "faiss": faiss.read_index("faiss_small.index"),
        "embed": SentenceTransformer("BAAI/bge-small-en-v1.5"),
        "rerank": CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "bm25": BM25Okapi([tokenize(c["text"]) for c in chunks]),
        "gemini": genai.Client(api_key=st.secrets["GEMINI_API_KEY"]),
        "tavily": TavilyClient(api_key=st.secrets["TAVILY_API_KEY"]),
        "web_cache": {}, "answer_cache": {}, "hits": [],
    }


R = load_resources()
chunks = R["chunks"]


# ---------------------------------------------------------------- document search
def hybrid_search(question, top_k=20):
    q_vec = R["embed"].encode(
        ["Represent this sentence for searching relevant passages: " + question],
        normalize_embeddings=True).astype("float32")
    _, faiss_ids = R["faiss"].search(q_vec, top_k)
    bm25_ids = np.argsort(R["bm25"].get_scores(tokenize(question)))[::-1][:top_k]

    fused = {}
    for ranked_list in (faiss_ids[0], bm25_ids):
        for rank, idx in enumerate(ranked_list):
            fused[int(idx)] = fused.get(int(idx), 0) + 1 / (60 + rank)
    return sorted(fused, key=fused.get, reverse=True)[:top_k]


def search_documents(question, final_k=4):
    candidates = hybrid_search(question)
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


def gather_evidence(question):
    doc_results = [r for r in search_documents(question) if r["score"] >= RELEVANCE_THRESHOLD]
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


def ask(question):
    cache = R["answer_cache"]
    if question in cache:
        return cache[question]
    ev = gather_evidence(question)
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
        return "I'm doing well, thank you! Ask me anything about the legal documents or Indian law."
    if re.fullmatch(r"(good\s*)?morning( there)?", t):
        return "Good morning! 👋 How can I help you with legal documents today?"
    if re.fullmatch(r"(good\s*)?afternoon( there)?", t):
        return "Good afternoon! 👋 What would you like to know?"
    if re.fullmatch(r"(good\s*)?evening( there)?", t):
        return "Good evening! 👋 What would you like to know?"
    if re.fullmatch(r"(good\s*)?night", t):
        return "Good night! Come back anytime you have a legal question."
    if re.fullmatch(r"(h+i+|h+e+l+l*o+|h+e+y+|namaste|hola)( there| lexivault)?", t):
        return "Hello! 👋 I'm LexiVault. Ask me about the document library or general Indian law."
    if re.search(r"\b(thanks|thank you|thx)\b", t) and len(t.split()) <= 5:
        return "You're welcome! Let me know if you have another question."
    if re.fullmatch(r"(bye|goodbye|see you|see ya)( later)?", t):
        return "Goodbye! Take care. 👋"
    if re.search(r"who are you|what can you do|^help$", t):
        return ("I'm LexiVault, a legal document assistant. I answer from the PDFs in my library and from "
                "trusted Indian legal websites, and I show the sources for every answer. If I can't find "
                "reliable evidence, I'll tell you.")
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


st.title("⚖️ LexiVault")
st.caption("Legal document question answering with cited sources. "
           "General information only, not legal advice.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.asked = 0

with st.sidebar:
    st.header("Try asking")
    for example in ["What is the earnest money deposit?",
                    "What is a mortgage deed?",
                    "What are the conditions for a Hindu marriage?",
                    "Can a landlord evict a tenant under the rent control act?",
                    "What is Section 420 of the IPC?",
                    "Is the EMD in my tender consistent with general financial rules in India?"]:
        if st.button(example, use_container_width=True):
            st.session_state.pending = example
    st.caption("Library: a government tender, a property documents guide, "
               "the Telangana rent control Act and the Hindu Marriage Act.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        show_message(m)

question = st.chat_input("Ask about the documents or Indian law...") or st.session_state.pop("pending", None)

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
                out = ask(question.strip())
            R["hits"].append(time.time())
            st.session_state.asked += 1
            msg = {"role": "assistant", "content": out["answer"], "sources": out["sources"]}
        show_message(msg)
    st.session_state.messages.append(msg)
