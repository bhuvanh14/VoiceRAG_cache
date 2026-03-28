"""
rag/knowledge_base.py
----------------------
ChromaDB knowledge base with domain switching support.

Three built-in domains:
  healthcare — medications, prescriptions, diabetes, appointments
  hr         — leave policy, onboarding, benefits, payroll
  legal      — NDAs, IP, contracts, compliance

HOW TO RUN:
    python -m rag.knowledge_base --sample       # index default (healthcare)
    python -m rag.knowledge_base --domain hr    # index HR domain
"""

import os
import json
import argparse
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger

CHROMA_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_WORDS = 256
OVERLAP     = 30

# ── Domain sample documents ──────────────────────────────────────────────────
DOMAIN_DOCS = {
    "healthcare": [
        ("med_1", "Hypertension, or high blood pressure, is treated using several drug classes. "
         "ACE inhibitors such as lisinopril relax blood vessels by blocking angiotensin. "
         "Beta-blockers like metoprolol reduce heart rate and cardiac output. "
         "Calcium channel blockers and diuretics are also commonly prescribed. "
         "Lifestyle changes including reduced sodium intake, regular exercise, and weight "
         "management complement drug therapy significantly."),
        ("med_2", "Prescription refills can be requested at any pharmacy using your prescription number. "
         "Most pharmacies offer automatic refill programs that notify you when medication is ready. "
         "You can also request refills through your doctor's patient portal or by calling the clinic. "
         "Insurance plans typically cover 30 or 90 day supplies. Emergency refills for controlled "
         "substances require a new prescription from your physician."),
        ("med_3", "Diabetes management involves monitoring blood glucose levels regularly, taking insulin "
         "or oral medications like metformin, and following a balanced diet low in refined carbohydrates. "
         "Regular HbA1c tests every 3 months track long-term glucose control. Exercise improves insulin "
         "sensitivity. Type 1 diabetes always requires insulin. Type 2 can often be managed with lifestyle."),
        ("med_4", "Common side effects of ACE inhibitors include a dry persistent cough, dizziness, "
         "and elevated potassium levels. Beta-blockers may cause fatigue, cold hands and feet, and "
         "can mask hypoglycemia symptoms in diabetic patients. Calcium channel blockers can cause "
         "ankle swelling and constipation. Always consult your physician before stopping any medication."),
        ("med_5", "Hospital appointments can be scheduled through the patient portal, by calling the "
         "clinic, or through your insurance provider's referral system. Bring your insurance card, "
         "photo ID, and a list of current medications to every appointment. Arrive 15 minutes early "
         "for new patient visits. Telemedicine appointments are available for follow-up consultations."),
        ("med_6", "Drug interactions are an important safety concern. NSAIDs like ibuprofen can reduce "
         "the effectiveness of ACE inhibitors and cause kidney problems. Statins combined with certain "
         "antibiotics can increase the risk of muscle damage. Always inform your pharmacist of all "
         "medications, supplements, and herbal remedies you are taking to avoid dangerous combinations."),
    ],
    "hr": [
        ("hr_1", "Employee onboarding typically covers company policy review, IT account setup, benefits "
         "enrollment, and team introductions. The standard probation period is 90 days. Health insurance "
         "and 401k matching become active after 30 days of employment. New employees should complete all "
         "onboarding paperwork within their first week including tax forms and direct deposit setup."),
        ("hr_2", "To request time off, submit a leave request through the HR portal at least 5 business "
         "days in advance for planned leave. Sick leave does not require advance notice but you must "
         "notify your manager by 9am on the day of absence. Annual leave accrues at 1.5 days per month. "
         "Unused leave up to 10 days can be carried over to the next calendar year."),
        ("hr_3", "The performance review cycle runs twice per year — mid-year in June and annual in "
         "December. Employees receive a self-assessment form two weeks before the review. Ratings are "
         "on a 5-point scale. Salary adjustments linked to performance take effect from January 1st. "
         "Promotion recommendations must be submitted by managers before November 30th."),
        ("hr_4", "Remote work policy allows employees to work from home up to 3 days per week with "
         "manager approval. Core hours of 10am to 3pm in the local time zone must be observed during "
         "remote days. A stable internet connection and a dedicated workspace are required. Equipment "
         "loans including laptops and monitors can be arranged through IT for approved remote workers."),
        ("hr_5", "Payroll is processed on the 15th and last working day of each month. Direct deposit "
         "is mandatory for all employees. Pay stubs are available through the HR portal. Overtime is "
         "paid at 1.5x the base hourly rate for hours exceeding 40 per week. Expense reimbursements "
         "are processed within 5 business days of submission with valid receipts."),
        ("hr_6", "The company offers a comprehensive benefits package including health, dental, and vision "
         "insurance. The 401k plan includes a 4% employer match for contributions up to 6% of salary. "
         "Employee assistance programs provide free confidential counseling. Tuition reimbursement of "
         "up to $5000 per year is available for role-related courses approved in advance by HR."),
    ],
    "legal": [
        ("leg_1", "A non-disclosure agreement or NDA is a contract establishing confidentiality between "
         "parties. It defines what information is considered confidential, the obligations of the "
         "receiving party, and the duration of the agreement. NDAs can be unilateral where only one "
         "party shares information or mutual where both parties share sensitive information. Breach of "
         "an NDA can result in injunctive relief and significant financial damages."),
        ("leg_2", "Intellectual property rights include patents, trademarks, copyrights, and trade secrets. "
         "Patents protect inventions for 20 years from the filing date. Copyrights protect original "
         "creative works and arise automatically upon creation without registration. Trademarks protect "
         "brand identifiers including names, logos, and slogans and can be renewed indefinitely. Trade "
         "secrets are protected as long as reasonable measures are taken to keep them confidential."),
        ("leg_3", "Contract formation requires offer, acceptance, and consideration. An offer is a clear "
         "proposal of terms. Acceptance must mirror the offer exactly — any changes constitute a "
         "counter-offer. Consideration is the value exchanged, which can be money, services, or a "
         "promise to act or refrain from acting. Contracts without consideration are generally not "
         "enforceable. Written contracts are strongly preferred for transactions over $500."),
        ("leg_4", "Employment contracts typically include terms on compensation, duties, working hours, "
         "confidentiality obligations, non-compete clauses, and termination conditions. Non-compete "
         "clauses must be reasonable in scope, geography, and duration to be enforceable. Most "
         "jurisdictions limit non-competes to 12 months and require geographic restrictions. Always "
         "have an employment contract reviewed by a qualified attorney before signing."),
        ("leg_5", "Data protection regulations such as GDPR in Europe and CCPA in California require "
         "organizations to protect personal data and respect individual rights. Organizations must "
         "obtain clear consent before collecting personal data, maintain records of data processing, "
         "and implement appropriate security measures. Individuals have the right to access, correct, "
         "and delete their personal data. Non-compliance can result in significant regulatory fines."),
        ("leg_6", "Contract disputes can be resolved through negotiation, mediation, arbitration, or "
         "litigation. Mediation is a voluntary non-binding process where a neutral mediator helps "
         "parties reach agreement. Arbitration is typically binding and faster and cheaper than court. "
         "Many commercial contracts include mandatory arbitration clauses. Litigation should generally "
         "be a last resort given its cost and time requirements."),
    ],
}


def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        chunks.append(" ".join(words[start: start + CHUNK_WORDS]))
        start += CHUNK_WORDS - OVERLAP
    return chunks


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

    def index_domain(self, domain: str):
        """Index built-in domain documents. Clears existing entries first."""
        docs_list = DOMAIN_DOCS.get(domain)
        if not docs_list:
            logger.warning(f"Unknown domain: {domain}")
            return

        # Clear existing entries
        try:
            existing = self.col.get()
            if existing["ids"]:
                self.col.delete(ids=existing["ids"])
        except Exception:
            pass

        docs, ids, metas = [], [], []
        for doc_id, text in docs_list:
            for i, chunk in enumerate(chunk_text(text)):
                docs.append(chunk)
                ids.append(f"{doc_id}_{i}")
                metas.append({"source": doc_id, "domain": domain})

        self.col.upsert(documents=docs, ids=ids, metadatas=metas)
        logger.info(f"Indexed {len(docs)} chunks for domain '{domain}'")

    def index_sample(self):
        """Index default healthcare sample (backward compatibility)."""
        self.index_domain("healthcare")

    def index_squad(self, path: str, max_articles: int = 2000):
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

    def index_text_dir(self, directory: str):
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

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.col.query(
            query_texts=[query],
            n_results=min(top_k, max(1, self.col.count()))
        )
        passages = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            passages.append({"text": doc, "source": meta.get("source", ""), "distance": dist})
        return passages

    def count(self) -> int:
        return self.col.count()


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample",  action="store_true")
    parser.add_argument("--domain",  type=str, choices=["healthcare","hr","legal"])
    parser.add_argument("--squad",   type=str)
    parser.add_argument("--textdir", type=str)
    args = parser.parse_args()

    domain = args.domain or "healthcare"
    kb = KnowledgeBase(collection_name=f"kb_{domain}")

    if args.sample or (not args.squad and not args.textdir):
        print(f"\nIndexing {domain} domain...")
        kb.index_domain(domain)
    if args.squad:
        kb.index_squad(args.squad)
    if args.textdir:
        kb.index_text_dir(args.textdir)

    print(f"Total chunks: {kb.count()}")
    results = kb.retrieve("What is the main topic of this knowledge base?", top_k=2)
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['text'][:100]}...")
    print("\n✅ Knowledge base ready!")