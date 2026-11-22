"""Evaluation harness for the GDPR chatbot.

Run from the project folder (gdpr-rag), with Ollama running and the index built:

    python evaluation/run_eval.py --mode retrieval              # fast, no LLM: did we fetch the right articles?
    python evaluation/run_eval.py --mode full --name plain_rag  # full answers: facts, citations, refusals, latency

Results are saved to evaluation/results/<name>.json. Re-run with a different --name after every change
(for example --name ontology_rag in Phase 3) and compare the tables.

Metrics
  hit@k        : at least one expected article is among the k retrieved chunks
  recall@k     : fraction of the expected articles that were retrieved (matters for multi-article questions)
  MRR          : 1 / rank of the first retrieved chunk from an expected article
  fact_cov     : fraction of key facts that appear in the answer
  cite_ok      : the answer cites at least one expected article
  abstain_ok   : out-of-scope questions get "does not cover this"; answerable questions do not
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB_DIR = ROOT / "chroma_db"
QUESTIONS = HERE / "eval_questions.json"
RESULTS = HERE / "results"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
ABSTAIN_PHRASE = "does not cover this"

# Keep this identical to SYSTEM in app.py so the numbers describe the real chatbot.
SYSTEM = (
    "You are a GDPR study assistant. Answer ONLY from the CONTEXT below. "
    "Cite every claim with the article label exactly as shown, e.g. [Art. 17]. "
    "If the context does not contain the answer, reply exactly: "
    "\"The provided GDPR text does not cover this.\" "
    "Keep the answer under 180 words, in simple English. "
    "This is educational information, not legal advice."
)


# ----------------------------------------------------------------------------- pure helpers (no Ollama needed)
def squash(s: str) -> str:
    """Lower-case and drop all whitespace so '4 %' matches '4%' and non-breaking spaces do not matter."""
    return re.sub(r"\s+", "", s.replace("\xa0", " ")).lower()


def retrieval_scores(expected, retrieved_articles):
    """expected: list[int]; retrieved_articles: list[int] in rank order (0 = recital chunk)."""
    exp = set(expected)
    got = set(retrieved_articles)
    hit = bool(exp & got)
    recall = len(exp & got) / len(exp) if exp else None
    rank = next((i + 1 for i, a in enumerate(retrieved_articles) if a in exp), None)
    return {"hit": hit, "recall": recall, "rr": (1 / rank) if rank else 0.0}


def cited_articles(answer: str):
    nums = re.findall(r"\[\s*Art\.?\s*(\d{1,3})", answer, flags=re.I)
    nums += re.findall(r"\bArticle\s+(\d{1,3})\b", answer, flags=re.I)
    return sorted({int(n) for n in nums})


def answer_scores(q, answer: str):
    abstained = ABSTAIN_PHRASE in answer.lower()
    if q["type"] == "out_of_scope":
        return {"abstain_ok": abstained, "fact_cov": None, "cite_ok": None, "cited": cited_articles(answer)}
    facts = q["key_facts"]
    cov = sum(squash(f) in squash(answer) for f in facts) / len(facts) if facts else None
    cited = cited_articles(answer)
    return {
        "abstain_ok": not abstained,
        "fact_cov": cov,
        "cite_ok": bool(set(cited) & set(q["expected_articles"])),
        "cited": cited,
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


# ----------------------------------------------------------------------------- pipeline (needs Ollama + ChromaDB)
def retrieve(col, question, k):
    import ollama
    emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"][0]
    res = col.query(query_embeddings=[emb], n_results=k)
    return res["documents"][0], res["metadatas"][0]


def generate(question, docs, metas):
    import ollama
    context = "\n\n".join(f"[{m['label']}] {d}" for d, m in zip(docs, metas))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]
    return ollama.chat(model=LLM_MODEL, messages=msgs, options={"temperature": 0})["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrieval", "full"], default="retrieval")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--name", default="plain_rag")
    args = ap.parse_args()

    import chromadb
    col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection("gdpr")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []

    for q in questions:
        t0 = time.time()
        docs, metas = retrieve(col, q["question"], args.k)
        row = {"id": q["id"], "type": q["type"], "question": q["question"],
               "expected": q["expected_articles"], "retrieved": [m["article"] for m in metas]}
        if q["type"] != "out_of_scope":
            row.update(retrieval_scores(q["expected_articles"], row["retrieved"]))
        if args.mode == "full":
            ans = generate(q["question"], docs, metas)
            row["answer"] = ans
            row.update(answer_scores(q, ans))
        row["latency_s"] = round(time.time() - t0, 2)
        rows.append(row)
        flag = "" if row.get("hit", True) else "  <-- expected article NOT retrieved"
        print(f"{q['id']} [{q['type']:<12}] retrieved arts {row['retrieved']} expected {q['expected_articles']}{flag}")

    ans_rows = [r for r in rows if r["type"] != "out_of_scope"]
    oos_rows = [r for r in rows if r["type"] == "out_of_scope"]
    rel_rows = [r for r in rows if r["type"] == "relational"]
    summary = {
        "name": args.name, "mode": args.mode, "k": args.k, "llm": LLM_MODEL, "embedding": EMBED_MODEL,
        "n_questions": len(rows),
        "hit@k": mean([float(r["hit"]) for r in ans_rows]),
        "recall@k": mean([r["recall"] for r in ans_rows]),
        "recall@k_relational": mean([r["recall"] for r in rel_rows]),
        "MRR": mean([r["rr"] for r in ans_rows]),
        "mean_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 2),
    }
    if args.mode == "full":
        summary.update({
            "fact_coverage": mean([r["fact_cov"] for r in ans_rows]),
            "citation_ok": mean([float(r["cite_ok"]) for r in ans_rows]),
            "answers_not_refused": mean([float(r["abstain_ok"]) for r in ans_rows]),
            "out_of_scope_refused": mean([float(r["abstain_ok"]) for r in oos_rows]),
        })

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>24}: {v}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
