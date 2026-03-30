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
    from pathlib import Path
    config_path = Path(config_path)
    if not config_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / config_path
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
    """Initialize the LLM — simplified to use Ollama only."""
    cfg = load_config(config_path)
    rag_cfg = cfg.get("rag", {})

    from langchain_community.llms import Ollama

    model = os.getenv("OLLAMA_MODEL", rag_cfg.get("ollama_model", "llama3.2:3b"))
    temperature = rag_cfg.get("temperature", 0.2)
    # Basic availability check: prefer the `ollama` CLI, but fall back to the
    # Python `ollama` client if available. Match model names using substring
    # membership to tolerate slight version differences (e.g. 'llama3.1').
    available_models = None
    model_candidates = []
    try:
        import subprocess

        proc = subprocess.run(["ollama", "ls"], capture_output=True, text=True, timeout=5)
        stdout = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0:
            # Collect lines and parse model names (first column is NAME).
            raw_lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
            cli_model_names = [ln.split()[0] for ln in raw_lines if ln.split()]
            model_candidates.extend(cli_model_names)
            available_models = "\n".join(cli_model_names)
            # don't fail yet; we'll try python client and also allow substring matching
        else:
            raise RuntimeError("ollama-cli-failed")
    except FileNotFoundError:
        # CLI not installed; we'll try the Python client next
        available_models = None
    except subprocess.SubprocessError:
        available_models = None
    except RuntimeError as e:
        if str(e) != "model-not-found-via-cli":
            available_models = None

    # If CLI didn't locate the model, try the Python `ollama` client (if installed)
    if available_models is None or (available_models is not None and model not in available_models):
        try:
            from ollama import Client

            client = Client()
            # Try a couple of possible attribute names used by different client versions
            models_list = None
            if hasattr(client, "list_models"):
                models_list = client.list_models()
            elif hasattr(client, "models"):
                models_list = client.models()
            elif hasattr(client, "get_models"):
                models_list = client.get_models()

            if models_list is not None:
                # Normalize to strings and join for reporting
                try:
                    model_names = [m["name"] if isinstance(m, dict) and "name" in m else str(m) for m in models_list]
                except Exception:
                    model_names = [str(m) for m in models_list]
                model_candidates.extend(model_names)
                available_models = "\n".join(model_candidates)
        except Exception:
            # If this fails, available_models may still be None or from the CLI attempt
            pass

    # If we have candidates, try to pick the best match (substring). If the
    # requested model is a prefix like 'llama3' and we have 'llama3.2:3b', choose that.
    if model_candidates:
        chosen = None
        for cand in model_candidates:
            if model == cand or model in cand:
                chosen = cand
                break
        if chosen:
            model = chosen
            available_models = "\n".join(model_candidates)
        else:
            avail_msg = f"\nAvailable models:\n{available_models}" if available_models else ""
            raise RuntimeError(
                f"Ollama model '{model}' not found locally. Pull it with `ollama pull {model}` or set `OLLAMA_MODEL` to a model you have installed.{avail_msg}"
            )

    return Ollama(model=model, temperature=temperature)


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
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            # If Ollama returns a 404 (model not found), raise a clear error telling the user
            try:
                from langchain_community.llms.ollama import OllamaEndpointNotFoundError
            except Exception:
                OllamaEndpointNotFoundError = None

            if OllamaEndpointNotFoundError is not None and isinstance(e, OllamaEndpointNotFoundError):
                raise RuntimeError(
                    "Ollama model not found (404). Pull the model locally with `ollama pull <model>` "
                    "or set the `OLLAMA_MODEL` environment variable to a model you have available."
                ) from e
            raise

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
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            try:
                from langchain_community.llms.ollama import OllamaEndpointNotFoundError
            except Exception:
                OllamaEndpointNotFoundError = None

            if OllamaEndpointNotFoundError is not None and isinstance(e, OllamaEndpointNotFoundError):
                raise RuntimeError(
                    "Ollama model not found (404). Pull the model locally with `ollama pull <model>` "
                    "or set the `OLLAMA_MODEL` environment variable to a model you have available."
                ) from e
            raise

        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "retrieved_docs": results,
            "query": question,
        }
