"""
RAG retrieval chain for industrial fault diagnosis.
Combines vector store retrieval with LLM generation.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


DIAGNOSIS_PROMPT_TEMPLATE = """You are an expert industrial maintenance engineer specializing in rolling bearing fault diagnosis.

Use the following context from the bearing maintenance manual and fault case database to answer the question.
If the context doesn't contain enough information, use your general knowledge about bearing faults, but indicate this clearly.

=== RETRIEVED CONTEXT ===
{context}
=========================

Question: {question}

Provide a structured response with:
1. **Diagnosis**: What fault is indicated and its severity
2. **Explanation**: Technical explanation of the fault mechanism
3. **Recommended Actions**: Step-by-step corrective actions
4. **Similar Cases**: Any related fault patterns to watch for

Answer:"""


QA_PROMPT_TEMPLATE = """You are an expert industrial maintenance engineer. Answer the question using the provided context from the rolling bearing handbook.

=== RETRIEVED CONTEXT ===
{context}
=========================

Question: {question}

Provide a clear, technically accurate answer. If the context doesn't cover the question, state that and provide what information you can from general bearing knowledge.

Answer:"""


def get_llm(config_path="configs/config.yaml"):
    """Initialize the LLM based on configuration."""
    cfg = load_config(config_path)
    rag_cfg = cfg["rag"]

    provider = os.getenv("LLM_PROVIDER", rag_cfg["llm_provider"])

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=rag_cfg["openai_model"],
            temperature=rag_cfg["temperature"],
        )
    elif provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=os.getenv("OLLAMA_MODEL", rag_cfg["ollama_model"]),
            temperature=rag_cfg["temperature"],
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


class RetrievalChain:
    """RAG chain that retrieves context and generates answers using an LLM."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def _format_context(self, results):
        """Format retrieved documents into a context string."""
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source_type", "unknown")
            if source == "manual":
                page = r["metadata"].get("page", "?")
                context_parts.append(f"[Manual, Page {page}]\n{r['text']}")
            elif source == "fault_case":
                fault = r["metadata"].get("fault_type", "?")
                context_parts.append(f"[Fault Case: {fault}]\n{r['text']}")
            else:
                context_parts.append(f"[Source {i}]\n{r['text']}")
        return "\n\n---\n\n".join(context_parts)

    def diagnose(self, query, fault_class=None, confidence=None):
        """Run diagnosis RAG chain with optional fault classification context."""
        # Build enriched query
        enriched_query = query
        if fault_class:
            enriched_query = (
                f"The CNN model classified this as: {fault_class}"
                + (f" (confidence: {confidence:.1%})" if confidence else "")
                + f"\n\nUser question: {query}"
            )

        results = self.retriever.retrieve(enriched_query)
        context = self._format_context(results)
        prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(context=context, question=enriched_query)
        response = self.llm.invoke(prompt)

        # Handle both string and AIMessage responses
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "retrieved_docs": results,
            "query": enriched_query,
        }

    def ask(self, question):
        """General Q&A against the manual knowledge base."""
        results = self.retriever.retrieve(question, source_type="manual")
        context = self._format_context(results)
        prompt = QA_PROMPT_TEMPLATE.format(context=context, question=question)
        response = self.llm.invoke(prompt)

        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "retrieved_docs": results,
            "query": question,
        }
