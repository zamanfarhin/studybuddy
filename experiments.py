"""
Experiment tracking for StudyBuddy.
Runs fixed test questions across configurations and records metrics.
"""

import time
import csv
import json
from openai import OpenAI, APITimeoutError
from rag import RAGEngine

TEST_QUESTIONS = [
    # Memory questions - general knowledge
    {"question": "What is machine learning?", "expected_type": "memory"},
    {"question": "Explain the concept of photosynthesis.", "expected_type": "memory"},
    {"question": "What is the Pythagorean theorem?", "expected_type": "memory"},
    {"question": "How does the internet work?", "expected_type": "memory"},
    # Retrieve questions - document-specific
    {"question": "What does the document say about the main topic?", "expected_type": "retrieve"},
    {"question": "According to the PDF, what are the key findings or main ideas?", "expected_type": "retrieve"},
    {"question": "Based on what I uploaded, summarize the main points.", "expected_type": "retrieve"},
    # Hybrid questions
    {"question": "How does the content in my document relate to broader scientific principles?", "expected_type": "hybrid"},
    {"question": "Explain the concepts from the uploaded document in a broader context.", "expected_type": "hybrid"},
    {"question": "Compare what my document says to general knowledge on the topic.", "expected_type": "hybrid"},
]

CHUNK_SIZES = [256, 512, 1024]
TOP_K_VALUES = [1, 3, 5]
MODELS = ["gpt-4o-mini", "gpt-4o"]
RESULTS_FILE = "experiments_results.csv"
TOTAL_RUNS = len(CHUNK_SIZES) * len(TOP_K_VALUES) * len(MODELS) * len(TEST_QUESTIONS)


class ExperimentRunner:
    def __init__(self, openai_api_key: str, pdf_bytes: bytes, pdf_name: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key, timeout=60.0)
        self.pdf_bytes = pdf_bytes
        self.pdf_name = pdf_name

    def _create_rag_engine(self, chunk_size: int) -> RAGEngine:
        """Create a RAG engine with the given chunk size and index the PDF."""
        persist_dir = f"./exp_chroma_{chunk_size}"
        print(f"[Experiment] Creating RAG engine: chunk_size={chunk_size}, dir={persist_dir}", flush=True)
        engine = RAGEngine(self.openai_api_key, persist_directory=persist_dir)
        engine.chunk_size = chunk_size
        print(f"[Experiment] Clearing existing collection for chunk_size={chunk_size}...", flush=True)
        engine.clear_collection()
        print(f"[Experiment] Indexing PDF with chunk_size={chunk_size}...", flush=True)
        num_chunks = engine.process_and_store_pdf(self.pdf_bytes, self.pdf_name)
        print(f"[Experiment] Indexed {num_chunks} chunks for chunk_size={chunk_size}.", flush=True)
        return engine

    def _decide_action(self, question: str, has_documents: bool, model: str) -> str:
        """Classify the query action using function calling."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify_query",
                    "description": "Classify the user's query to determine the best way to answer it",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reasoning": {
                                "type": "string",
                                "description": "Brief reasoning for this classification"
                            },
                            "action": {
                                "type": "string",
                                "enum": ["memory", "retrieve", "hybrid"],
                                "description": "memory for general knowledge, retrieve for document-specific, hybrid for both"
                            }
                        },
                        "required": ["reasoning", "action"]
                    }
                }
            }
        ]

        classification_prompt = f"""You are a query classifier. Analyze this question and decide HOW to answer it.

USER QUESTION: "{question}"

DOCUMENTS UPLOADED: {has_documents}

CLASSIFICATION RULES:
1. **memory** - General knowledge questions (definitions, concepts, common facts)
2. **retrieve** - ONLY when the question explicitly references uploaded documents
3. **hybrid** - When both general knowledge and document context are needed

Call the classify_query function with your reasoning and decision."""

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": classification_prompt}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "classify_query"}},
            temperature=0,
            timeout=60.0
        )

        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        action = result.get("action", "memory")

        if action not in ["memory", "retrieve", "hybrid"]:
            action = "memory"
        if not has_documents and action in ["retrieve", "hybrid"]:
            action = "memory"

        return action

    def _generate_response(self, question: str, action: str, rag_engine: RAGEngine,
                            top_k: int, model: str) -> str:
        """Generate a response for the experiment."""
        context = ""
        if action in ["retrieve", "hybrid"]:
            retrieved_docs = rag_engine.retrieve(question, n_results=top_k)
            if retrieved_docs:
                context = "Relevant information from uploaded documents:\n\n"
                for doc in retrieved_docs:
                    context += f"[Source: {doc['source']}]\n{doc['content']}\n\n"

        if action == "memory":
            system_prompt = "You are StudyBuddy, a helpful study assistant. Answer using your general knowledge."
        elif action == "retrieve":
            system_prompt = f"You are StudyBuddy, a helpful study assistant. Answer based on the following context.\n\n{context}"
        else:
            system_prompt = f"You are StudyBuddy, a helpful study assistant. Answer using both general knowledge and this context.\n\n{context}"

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            timeout=60.0
        )

        return response.choices[0].message.content

    def run_all(self, progress_callback=None) -> list[dict]:
        """Run all experiments across all configurations and save to CSV."""
        results = []
        done = 0

        # Build one engine per chunk size (re-indexes the PDF at each chunk size)
        print(f"[Experiment] Starting indexing phase: {len(CHUNK_SIZES)} chunk sizes to index.", flush=True)
        engines = {}
        for idx, chunk_size in enumerate(CHUNK_SIZES):
            print(f"[Experiment] Indexing {idx + 1}/{len(CHUNK_SIZES)}: chunk_size={chunk_size}...", flush=True)
            engines[chunk_size] = self._create_rag_engine(chunk_size)
        print(f"[Experiment] Indexing complete. Starting {TOTAL_RUNS} experiment runs...", flush=True)

        for chunk_size in CHUNK_SIZES:
            engine = engines[chunk_size]
            has_docs = engine.get_document_count() > 0

            for top_k in TOP_K_VALUES:
                for model in MODELS:
                    for q in TEST_QUESTIONS:
                        question = q["question"]
                        print(
                            f"[Experiment] Run {done + 1}/{TOTAL_RUNS} | "
                            f"chunk={chunk_size} top_k={top_k} model={model} | "
                            f"Q: {question[:50]}...",
                            flush=True
                        )
                        start = time.time()
                        try:
                            print(f"[Experiment]   -> classify_query...", flush=True)
                            action = self._decide_action(question, has_docs, model)
                            print(f"[Experiment]   -> action={action}, generating response...", flush=True)
                            response = self._generate_response(question, action, engine, top_k, model)
                            elapsed = round(time.time() - start, 2)
                            print(f"[Experiment]   -> done in {elapsed}s", flush=True)
                            results.append({
                                "chunk_size": chunk_size,
                                "top_k": top_k,
                                "model": model,
                                "question": question[:60] + "..." if len(question) > 60 else question,
                                "expected_type": q["expected_type"],
                                "decision": action,
                                "response_time_s": elapsed,
                                "response_length": len(response)
                            })
                        except APITimeoutError as e:
                            print(f"[Experiment]   -> TIMEOUT: {e}", flush=True)
                            results.append({
                                "chunk_size": chunk_size,
                                "top_k": top_k,
                                "model": model,
                                "question": question[:60] + "..." if len(question) > 60 else question,
                                "expected_type": q["expected_type"],
                                "decision": "ERROR",
                                "response_time_s": 30.0,
                                "response_length": 0
                            })
                        except Exception as e:
                            print(f"[Experiment]   -> ERROR: {e}", flush=True)
                            results.append({
                                "chunk_size": chunk_size,
                                "top_k": top_k,
                                "model": model,
                                "question": question[:60] + "..." if len(question) > 60 else question,
                                "expected_type": q["expected_type"],
                                "decision": "ERROR",
                                "response_time_s": 30.0,
                                "response_length": 0
                            })

                        done += 1
                        if progress_callback:
                            progress_callback(done / TOTAL_RUNS)

        if results:
            fieldnames = list(results[0].keys())
            with open(RESULTS_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        return results
