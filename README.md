
# EvidenceRAG — A Research Assistant That Knows When It Doesn't Know

**🎥 Demo Video:** 

https://plakshauniversity1-my.sharepoint.com/:v:/g/personal/jyothirmai_p_pg26_plaksha_edu_in/IQBCQ3bhMT3tQJjciAoyyrLBAeXZOvJvoX9pNggw4WtYHNY?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=ofs9Lm


**Author:** Jyothirmai P  
**Project:** EvidenceRAG — Lab 6 Capstone

EvidenceRAG is a multi-paper research assistant that retrieves evidence from 21 research papers on LLM hallucination and factuality, then checks whether that evidence is sufficient before answering. It compares a standard RAG pipeline with an evidence-aware pipeline that can answer supported questions, refuse unsupported questions, or identify conflicting evidence.

## 1. What does this do?

EvidenceRAG investigates whether adding an **evidence-verification step** to a standard Retrieval-Augmented Generation (RAG) system can reduce unsupported answers.

The system uses 21 research papers about hallucination, factuality, truthfulness, and factual consistency. It:

1. Loads and cleans the research PDFs.
2. Splits the documents into overlapping chunks.
3. Creates embeddings using `all-MiniLM-L6-v2`.
4. Retrieves the top-5 chunks using cosine similarity.
5. Uses `google/flan-t5-base` to verify whether the retrieved evidence is:

   * `SUPPORTED`
   * `INSUFFICIENT`
   * `CONFLICTING`
6. Generates an answer only when the evidence is considered sufficient.
7. Refuses questions when the corpus does not provide sufficient evidence.
8. Compares the evidence-aware system with a basic RAG baseline.

The main research question is:

> **Can evidence-aware retrieval improve the reliability of a multi-paper RAG assistant by reducing unsupported answers without excessively reducing its ability to answer supported questions?**

---

## 2. How do I run it?

### Requirements

* Python 3.10+
* A CPU is sufficient, although model inference can take some time.
* Internet access is required the first time the Hugging Face models are downloaded.

### Project structure
```text
Evidence_RAG/
│
├── data/
│   └── [your 21 research PDFs]
│
├── experiments/
│   ├── basic_rag_results.json
│   ├── evidencerag_final_results.json
│   ├── evidencerag_results.json
│   ├── questions.json
│   ├── retrieval_results.json
│   ├── reference_aware_retrieval_results.json
│   └── reference_aware_retrieval_v2_results.json
│
├── src/
│   ├── cleaner.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── evaluate_results.py
│   ├── evidence_checker.py
│   ├── evidence_verifier.py
│   ├── generator.py
│   ├── inspect_failures.py
│   ├── pdf_loader.py
│   ├── preprocess.py
│   ├── retriever.py
│   ├── retriever_reference_aware.py
│   ├── threshold_experiment.py
│   └── demo.py
│
├── EvidenceRAG_Evaluation.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

### Installation

Clone the repository:

```bash
git clone <https://github.com/pjyo57/Evidence_RAG>
cd Evidence_RAG
```

Create a virtual environment:

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

### Run the final EvidenceRAG experiment

From the project root:

```powershell
python src\generator.py
```

The final pipeline loads the prepared chunks and embeddings, retrieves evidence for the 12 evaluation questions, verifies the evidence, generates answers or refusals, and saves the results to:

```text
experiments\evidencerag_final_results.json
```

### Evaluate the experiment

Run:

```powershell
python src\evaluate_results.py
```

This compares the Basic RAG and EvidenceRAG results and reports the main evaluation metrics.

---

## 3. What did you find?

The experiment evaluated **12 questions**:

* 10 answerable questions
* 2 deliberately unanswerable questions

The evaluation was designed to measure retrieval separately from generation and to test whether the evidence gate could prevent unsupported answers.

### Retrieval baseline

The retrieval baseline achieved:

**Retrieval@5 = 1.00**

for the 10 answerable questions.

This means that for every answerable question, at least one expected source document appeared among the five retrieved chunks.

However, retrieval success does **not** mean that the generated answer is correct. The experiment showed that high retrieval similarity alone was not sufficient to guarantee answerability.

For example:

| Question | Type         | Strongest similarity |
| -------- | ------------ | -------------------: |
| Q6       | Answerable   |               0.7072 |
| Q11      | Unanswerable |               0.7097 |

The unanswerable question therefore had a slightly **higher** similarity score than the answerable question. This showed that a simple similarity threshold could not reliably separate supported from unsupported questions.

### Basic RAG vs EvidenceRAG

| Metric                                        |  Basic RAG | EvidenceRAG |
| --------------------------------------------- | ---------: | ----------: |
| Unsupported answers on unanswerable questions | 2/2 (100%) |    0/2 (0%) |
| Correct refusals on unanswerable questions    |   0/2 (0%) |  2/2 (100%) |
| Answerable questions allowed through          |          — |  8/10 (80%) |
| False refusals on answerable questions        |          — |  2/10 (20%) |
| Conflicting cases detected                    |          — |           1 |

The most important result was the change in unsupported answers on the two unanswerable questions:

**Basic RAG: 2/2 unsupported answers (100%)**

**EvidenceRAG: 0/2 unsupported answers (0%)**

For example, Q11 asked:

> Which hallucination detection method is definitively the most accurate method for every possible large language model and every task?

Basic RAG produced an unsupported answer:

> `Detection Benchmarks.`

EvidenceRAG classified the evidence as `INSUFFICIENT` and refused to answer.

This demonstrates the main safety benefit of the evidence-aware approach.

### Trade-off

EvidenceRAG was also conservative.

It correctly refused both unanswerable questions, but it also refused two answerable questions involving more complex multi-paper synthesis or comparison (Q5 and Q9).

Therefore:

* Correct refusal rate on unanswerable questions: **100%**
* Unsupported-answer rate on unanswerable questions: **0%**
* Answerable-question coverage: **80%**
* False-refusal rate on answerable questions: **20%**

The 80% figure is **not treated as overall accuracy** because the experiment did not establish that every answer allowed through was fully correct.

### Negative results

Several interventions were tested and did not improve the system.

#### Similarity-threshold tuning

A range of evidence similarity thresholds was tested. Lower thresholds allowed more answerable questions through but failed to block the unanswerable questions reliably. Higher thresholds blocked more unanswerable content but also rejected more answerable questions.

The experiment therefore did not find a clean similarity threshold that separated supported from unsupported questions.

#### Question-type-aware verification

A second verifier experiment added the question type (`single_paper`, `multi_paper`, `comparison`, or `unanswerable`) to the verification prompt.

This did not improve reliability. In particular, the experimental version incorrectly allowed the two deliberately unanswerable questions through, so the original verifier was retained.

#### Reference-aware retrieval

A reference-aware retrieval experiment attempted to identify bibliography/reference chunks and penalize them during retrieval.

The heuristic was too conservative and failed to reliably identify obvious reference entries in the extracted PDF text. Therefore, this intervention was not used as evidence of improvement.

These negative results were retained because they show which approaches were tested and why they were not adopted.

---
## 4. What would you do next?

The next improvement would be **claim-level evidence verification**.

The current verifier evaluates whether the retrieved evidence is sufficient for the question as a whole. This is limiting for complex multi-paper and comparison questions, where different parts of the answer may require evidence from different papers.

A stronger version would:

1. Break a question into individual claims.
2. Retrieve evidence for each claim.
3. Check each claim against its supporting passages.
4. For comparison questions, verify both sides of the comparison separately.
5. Generate the final answer only from verified claims.

I would also improve PDF text extraction and reference detection because extracted papers sometimes contain formatting artifacts and bibliography text that can affect retrieval.

Finally, I would evaluate the system on a larger and more diverse set of answerable and unanswerable questions. The main goal would be to reduce the current **20% false-refusal rate** while preserving the observed **0% unsupported-answer rate** on the unanswerable test questions.
