"""
rag/knowledge_base.py
----------------------
ChromaDB knowledge base with Wikipedia-powered domain indexing.

On first run for each domain, fetches real Wikipedia articles.
Results are cached in ChromaDB so subsequent startups are instant.

Three domains with Wikipedia topics:
  healthcare — hypertension, diabetes, medications, prescriptions
  hr         — employment law, leave, payroll, performance reviews
  legal      — NDAs, contracts, IP, data protection

HOW TO RUN:
    python -m rag.knowledge_base --domain healthcare
    python -m rag.knowledge_base --domain hr
    python -m rag.knowledge_base --domain legal
"""

import os
import json
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 256
OVERLAP     = 30

# ── Wikipedia topics per domain ───────────────────────────────────────────────
DOMAIN_TOPICS = {
    "healthcare": [
        "Hypertension",
        "ACE_inhibitor",
        "Beta_blocker",
        "Calcium_channel_blocker",
        "Diuretic",
        "Diabetes_mellitus",
        "Metformin",
        "Insulin",
        "Blood_pressure",
        "Prescription_drug",
        "Drug_interaction",
        "Lisinopril",
        "Metoprolol",
        "Antihypertensive_drug",
        "Glycated_hemoglobin",
    ],
    "hr": [
        "Annual_leave",
        "Employment_contract",
        "Performance_appraisal",
        "Payroll",
        "Employee_benefits",
        "Remote_work",
        "Sick_leave",
        "Probation_(employment)",
        "Onboarding",
        "401(k)",
        "Health_insurance_in_the_United_States",
        "Non-compete_clause",
        "Overtime",
        "Tuition_assistance",
        "Direct_deposit",
    ],
    "legal": [
        "Non-disclosure_agreement",
        "Intellectual_property",
        "Patent",
        "Copyright",
        "Trademark",
        "Trade_secret",
        "Contract",
        "Consideration_in_contracts",
        "Arbitration",
        "Mediation",
        "General_Data_Protection_Regulation",
        "California_Consumer_Privacy_Act",
        "Employment_contract",
        "Breach_of_contract",
        "Injunction",
    ],
}

# Fallback sample docs if Wikipedia fetch fails
FALLBACK_DOCS = {
    "healthcare": [
        ("med_1", "Hypertension is treated with ACE inhibitors such as lisinopril, beta-blockers like "
         "metoprolol, calcium channel blockers, and diuretics. Lifestyle changes including reduced "
         "sodium intake and regular exercise complement drug therapy."),
        ("med_2", "Prescription refills can be requested at any pharmacy using your prescription number. "
         "Emergency refills for controlled substances require a new prescription from your physician."),
        ("med_3", "Diabetes management involves monitoring blood glucose, taking metformin or insulin, "
         "and following a low-carbohydrate diet. HbA1c tests every 3 months track long-term control."),
    ],
    "hr": [
        ("hr_1", "Annual leave accrues at 1.5 days per month. Submit leave requests through the HR portal "
         "at least 5 business days in advance. Unused leave up to 10 days can be carried over."),
        ("hr_2", "Remote work is allowed up to 3 days per week with manager approval. Core hours of "
         "10am to 3pm must be observed. Equipment loans can be arranged through IT."),
    ],
    "legal": [
        ("leg_1", "A non-disclosure agreement establishes confidentiality between parties. Breach can "
         "result in injunctive relief and financial damages. NDAs can be unilateral or mutual."),
        ("leg_2", "Patents protect inventions for 20 years. Copyrights arise automatically upon creation. "
         "Trademarks protect brand identifiers and can be renewed indefinitely."),
    ],
}


def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + CHUNK_WORDS]))
        start += CHUNK_WORDS - OVERLAP
    return [c for c in chunks if len(c.split()) > 20]  # skip very short chunks


def fetch_wikipedia(topic: str) -> str | None:
    """
    Fetch full Wikipedia article text for a topic.
    Uses the Wikipedia REST API — no API key needed, completely free.
    Returns the article text or None if not found.
    """
    # Try full article first via parse API
    encoded = urllib.parse.quote(topic)
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={encoded}&prop=extracts&explaintext=true"
        f"&exsectionformat=plain&format=json&redirects=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VoiceRAG-Cache/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data  = json.loads(r.read())
            pages = data["query"]["pages"]
            page  = next(iter(pages.values()))
            text  = page.get("extract", "")
            if text and len(text) > 200:
                logger.info(f"Wikipedia: fetched '{topic}' ({len(text.split())} words)")
                return text
    except Exception as e:
        logger.warning(f"Wikipedia full fetch failed for '{topic}': {e}")

    # Fallback to summary API
    try:
        summary_url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/"
            f"{encoded}"
        )
        req = urllib.request.Request(summary_url, headers={"User-Agent": "VoiceRAG-Cache/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            text = data.get("extract", "")
            if text:
                logger.info(f"Wikipedia summary: fetched '{topic}' ({len(text.split())} words)")
                return text
    except Exception as e:
        logger.warning(f"Wikipedia summary fetch failed for '{topic}': {e}")

    return None


class KnowledgeBase:
    def __init__(self, collection_name: str = "knowledge_base"):
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name
        logger.info(f"KnowledgeBase '{collection_name}' ready ✅ | {self.col.count()} chunks")

    def index_domain(self, domain: str, force_reindex: bool = False):
        """
        Index a domain using Wikipedia articles.
        If the collection already has content and force_reindex=False,
        skips indexing (uses cached ChromaDB data).
        """
        if self.col.count() > 0 and not force_reindex:
            logger.info(
                f"KnowledgeBase already has {self.col.count()} chunks — skipping reindex. "
                f"Pass force_reindex=True to rebuild."
            )
            return

        topics = DOMAIN_TOPICS.get(domain, [])
        if not topics:
            logger.warning(f"Unknown domain: {domain}")
            return

        logger.info(f"Indexing '{domain}' domain from Wikipedia ({len(topics)} topics)...")

        # Clear existing entries
        try:
            existing = self.col.get()
            if existing["ids"]:
                self.col.delete(ids=existing["ids"])
                logger.info("Cleared existing entries.")
        except Exception:
            pass

        docs, ids, metas = [], [], []
        successful = 0

        for topic in topics:
            text = fetch_wikipedia(topic)
            time.sleep(0.3)  # be polite to Wikipedia API

            if text:
                for i, chunk in enumerate(chunk_text(text)):
                    docs.append(chunk)
                    ids.append(f"wiki_{topic}_{i}")
                    metas.append({
                        "source": f"Wikipedia: {topic.replace('_', ' ')}",
                        "domain": domain,
                        "topic":  topic,
                    })
                successful += 1
            else:
                logger.warning(f"Could not fetch Wikipedia article for '{topic}' — skipping.")

        if not docs:
            logger.warning("No Wikipedia articles fetched — using fallback sample docs.")
            self._index_fallback(domain)
            return

        # Batch upsert into ChromaDB
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            self.col.upsert(
                documents=docs[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metas[i:i+batch_size],
            )

        logger.info(
            f"✅ Indexed {len(docs)} chunks from {successful}/{len(topics)} "
            f"Wikipedia articles for domain '{domain}'"
        )

    def _index_fallback(self, domain: str):
        """Use hardcoded fallback docs if Wikipedia is unavailable."""
        fallback = FALLBACK_DOCS.get(domain, [])
        docs, ids, metas = [], [], []
        for doc_id, text in fallback:
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(chunk)
                ids.append(f"{doc_id}_{i}")
                metas.append({"source": doc_id, "domain": domain})
        if docs:
            self.col.upsert(documents=docs, ids=ids, metadatas=metas)
            logger.info(f"Indexed {len(docs)} fallback chunks for '{domain}'")

    def index_wikipedia(self, topics: list[str], domain: str = "custom"):
        """Index any list of Wikipedia topics directly."""
        docs, ids, metas = [], [], []
        for topic in topics:
            text = fetch_wikipedia(topic)
            time.sleep(0.3)
            if text:
                for i, chunk in enumerate(chunk_text(text)):
                    docs.append(chunk)
                    ids.append(f"wiki_{topic}_{i}")
                    metas.append({"source": f"Wikipedia: {topic.replace('_',' ')}", "domain": domain})
        if docs:
            self.col.upsert(documents=docs, ids=ids, metadatas=metas)
            logger.info(f"Indexed {len(docs)} chunks from {len(topics)} Wikipedia topics")

    def index_text_dir(self, directory: str):
        """Index all .txt files in a directory."""
        docs, ids, metas = [], [], []
        for path in Path(directory).glob("**/*.txt"):
            text = path.read_text(errors="ignore")
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(chunk)
                ids.append(f"{path.stem}_{i}")
                metas.append({"source": str(path)})
        if docs:
            self.col.upsert(documents=docs, ids=ids, metadatas=metas)
            logger.info(f"Indexed {len(docs)} chunks from {directory}")

    def index_squad(self, path: str, max_articles: int = 2000):
        """Index SQuAD 2.0 dataset."""
        with open(path) as f:
            data = json.load(f)
        docs, ids, metas = [], [], []
        count = 0
        for article in data["data"]:
            for para in article["paragraphs"]:
                for i, chunk in enumerate(chunk_text(para["context"])):
                    docs.append(chunk)
                    ids.append(f"squad_{count}_{i}")
                    metas.append({"source": "squad", "title": article["title"]})
                count += 1
                if count >= max_articles:
                    break
            if count >= max_articles:
                break
        for i in range(0, len(docs), 500):
            self.col.upsert(documents=docs[i:i+500], ids=ids[i:i+500], metadatas=metas[i:i+500])
        logger.info(f"Indexed {len(docs)} SQuAD chunks.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k relevant chunks for a query."""
        n = min(top_k, max(1, self.col.count()))
        results = self.col.query(query_texts=[query], n_results=n)
        passages = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            passages.append({
                "text":   doc,
                "source": meta.get("source", ""),
                "distance": dist,
            })
        return passages

    def count(self) -> int:
        return self.col.count()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceRAG-Cache Knowledge Base Builder")
    parser.add_argument("--domain",  type=str, choices=["healthcare","hr","legal"], default="healthcare")
    parser.add_argument("--force",   action="store_true", help="Force reindex even if data exists")
    parser.add_argument("--squad",   type=str, help="Path to SQuAD train-v2.0.json")
    parser.add_argument("--textdir", type=str, help="Directory of .txt files to index")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"Building knowledge base — domain: {args.domain}")
    print(f"{'='*55}\n")

    kb = KnowledgeBase(collection_name=f"kb_{args.domain}")

    if args.squad:
        kb.index_squad(args.squad)
    elif args.textdir:
        kb.index_text_dir(args.textdir)
    else:
        kb.index_domain(args.domain, force_reindex=args.force)

    print(f"\nTotal chunks indexed: {kb.count()}")
    print("\nTest retrieval:")
    test_queries = {
        "healthcare": "What medications treat high blood pressure?",
        "hr":         "How do I request time off from work?",
        "legal":      "What is a non-disclosure agreement?",
    }
    results = kb.retrieve(test_queries.get(args.domain, "Tell me about this domain"), top_k=3)
    for r in results:
        print(f"  [{r['distance']:.3f}] [{r['source']}]")
        print(f"           {r['text'][:120]}...")
    print(f"\n✅ Knowledge base ready — {kb.count()} chunks from Wikipedia!")