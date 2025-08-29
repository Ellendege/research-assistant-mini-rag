# 📘 Research Assistant (Mini-RAG on PDF in Colab)

Turn any long PDF into **executive-style bullet summaries** with **retrieval + summarization**, fully in **Google Colab (free)**.

## 🎯 What this shows recruiters
- **AI Engineering:** PDF parsing → chunking (465 chunks) → **embeddings** (MiniLM-L6-v2) → **FAISS** vector search → **summarization** (BART).
- **Prompt Engineering:** XML structure, grounding rules, page-citation mindset.
- **Debugging:** Fixed token overflow (512), repetition loops, and “Not in document” false negatives by adjusting context and swapping models.
- **Outcome:** Clean, skim-friendly executive bullets that decision-makers love.

---

## 🧩 Architecture (mini-RAG)
PDF → pypdf (pages) → chunk(900/150 overlap)
→ SentenceTransformers embeddings (MiniLM-L6-v2)
→ FAISS index (cosine)
→ Retrieve top-k chunks for a query
→ Summarize with facebook/bart-large-cnn
→ Output: Executive bullets (optionally with [p. x] tags)


---

## 🚀 Quickstart (Colab)
1. Open **Google Colab** → New Notebook.
2. Install:
   ```python
   !pip -q install pypdf faiss-cpu sentence-transformers transformers accelerate einops

load models

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMB_MODEL)

GEN_MODEL = "facebook/bart-large-cnn"  # summarizer specialist
generator_tok = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
gen = pipeline("summarization", model=generator, tokenizer=generator_tok)

Upload your PDF
from google.colab import files
up = files.upload()
pdf_path = list(up.keys())[0]

Parse → chunk:
from pypdf import PdfReader; import re
def load_pdf_pages(path):
    r = PdfReader(path); pages=[]
    for i,p in enumerate(r.pages):
        t=(p.extract_text() or "")
        t=re.sub(r"\s+"," ",t).strip()
        if t: pages.append({"page":i+1,"text":t})
    return pages
pages = load_pdf_pages(pdf_path)

def chunk_text(text, page, size=900, overlap=150):
    out=[]; s=0
    while s < len(text):
        e=min(s+size,len(text))
        out.append({"page":page,"chunk":text[s:e]})
        if e==len(text): break
        s=e-overlap
    return out

docs=[]
for p in pages: docs += chunk_text(p["text"], p["page"])

Embeddings → FAISS:
import numpy as np, faiss
def embed_texts(texts):
    v = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return v.astype("float32")
corpus = [d["chunk"] for d in docs]
embs = embed_texts(corpus)
index = faiss.IndexFlatIP(embs.shape[1]); index.add(embs)

Retrieve → summarize:

def retrieve(q, k=6):
    qv = embed_texts([q]); D,I = index.search(qv,k)
    return [(i, D[0][j]) for j,i in enumerate(I[0])]
def build_context(q, k=6, max_chunks=5):
    hits = retrieve(q,k); blocks=[]
    for i,_ in hits[:max_chunks]:
        blocks.append(f"[p. {docs[i]['page']}] {docs[i]['chunk']}")
    return "\n\n".join(blocks)

ctx = build_context("perfectionism", k=8, max_chunks=5)
summary = gen(ctx, max_length=120, min_length=60, do_sample=False)[0]["summary_text"]
print(summary)


Sample Output
### Executive Summary
- When we expect that our work must be perfect the first time, we fall into a cycle of perfectionism.
- Perfectionism isn’t actually wanting everything to be right.
- It hinders growth by setting unrealistic expectations about what we’re capable of.



Tech Stack

Python, Google Colab

pypdf, sentence-transformers (MiniLM-L6-v2), faiss-cpu

transformers, facebook/bart-large-cnn


Troubleshooting Notes

Token overflow (512 limit) → reduce retrieved context; use shorter outputs.

Repeating outputs (“self-sabotage…”) → switch to facebook/bart-large-cnn.

“Not in document” everywhere → ask concrete topical queries (e.g., perfectionism), not meta prompts.

Structure
research-assistant-mini-rag/
├─ notebooks/
│  └─ mini_rag_colab.ipynb
├─ samples/
│  ├─ output_executive_summary.md
│  └─ used_context.txt
├─ README.md
└─ LICENSE



License
MIT

---


