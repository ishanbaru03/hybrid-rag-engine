import time
from typing import List, Optional
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError, ClientError
from app.core.config import settings


class GenAIService:
    _client = None

    PRIMARY_MODEL = "gemini-3.6-flash"
    FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]

    @classmethod
    def get_client(cls) -> genai.Client:
        if cls._client is None:
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not configured in .env")
            cls._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return cls._client

    @classmethod
    def generate_embedding(cls, text: str) -> List[float]:
        """Generates a 768-dim text embedding for a single query."""
        client = cls.get_client()
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return response.embeddings[0].values

    @classmethod
    def generate_embeddings_batch(cls, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings in safe batches of 25 chunks per API call.
        Includes automatic 429 quota backoff and retries.
        """
        if not texts:
            return []

        client = cls.get_client()
        all_embeddings: List[List[float]] = []
        batch_size = 25  # 100 chunks take only 4 API calls instead of 100!

        for i in range(0, len(texts), batch_size):
            chunk_batch = texts[i : i + batch_size]
            
            # Retry loop for 429 rate limit backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=chunk_batch,
                        config=types.EmbedContentConfig(output_dimensionality=768),
                    )
                    for item in response.embeddings:
                        all_embeddings.append(item.values)
                    
                    # Polite micro-sleep between batches to keep RPM well below limits
                    time.sleep(0.5)
                    break
                except ClientError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < max_retries - 1:
                            print(f"[PrepIQ RateLimit] Embedding quota hit. Pausing 55s before resuming batch {i}...")
                            time.sleep(55)
                            continue
                    raise e
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise e

        return all_embeddings

    @classmethod
    def _generate_with_fallback(cls, prompt: str, system_instruction: str, is_json: bool = False, temperature: float = 0.2) -> str:
        client = cls.get_client()
        models_to_try = [cls.PRIMARY_MODEL] + cls.FALLBACK_MODELS

        config_params = {
            "system_instruction": system_instruction,
            "temperature": temperature,
        }
        if is_json:
            config_params["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_params)

        last_error = None
        for model in models_to_try:
            for attempt in range(2):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    return response.text
                except (ServerError, APIError, ClientError) as e:
                    last_error = e
                    time.sleep(1)
                    continue
                except Exception as e:
                    last_error = e
                    break

        raise RuntimeError(f"All GenAI models currently unavailable: {last_error}")

    @classmethod
    def answer_question(cls, question: str, context: str) -> str:
        system_instruction = (
            "You are PrepIQ, an expert academic study assistant. "
            "Use ONLY the provided context from the student's study notes to answer the question clearly, concisely, and accurately. "
            "Highlight key definitions, formulas, or steps where applicable. "
            "If the answer cannot be found in the context, explicitly say that the notes do not contain this information."
        )

        prompt = (
            f"Context from Study Notes:\n{context}\n\n"
            f"Student Question: {question}\n\n"
            f"Detailed Answer:"
        )

        return cls._generate_with_fallback(prompt=prompt, system_instruction=system_instruction, temperature=0.2)

    @classmethod
    def generate_quiz(cls, context: str, num_questions: int = 5) -> str:
        system_instruction = (
            "You are an expert exam creator for computer science and engineering. "
            "Generate academic multiple-choice practice questions strictly based on the provided notes context. "
            "Return the response as a JSON array of objects with the following keys:\n"
            "- question (string)\n"
            "- options (list of 4 strings: A, B, C, D)\n"
            "- correct_answer (string matching one of the options)\n"
            "- explanation (string explaining why this is correct based on the notes)"
        )

        prompt = (
            f"Generate {num_questions} conceptual and algorithmic practice questions from these notes:\n\n"
            f"{context}\n\n"
            f"Respond ONLY with valid JSON."
        )

        return cls._generate_with_fallback(prompt=prompt, system_instruction=system_instruction, is_json=True, temperature=0.3)

    @classmethod
    def generate_flashcards(cls, context: str, count: int = 6) -> str:
        system_instruction = (
            "You are an academic study coach. Create active-recall revision flashcards based on the notes. "
            "Each flashcard must have a 'front' (term, formula name, or core question) and 'back' (precise definition, formula, or breakdown). "
            "Return ONLY a JSON array of objects with keys: 'front' and 'back'."
        )

        prompt = (
            f"Create {count} high-yield revision flashcards from the following notes:\n\n"
            f"{context}\n\n"
            f"Respond ONLY with valid JSON."
        )

        return cls._generate_with_fallback(prompt=prompt, system_instruction=system_instruction, is_json=True, temperature=0.3)

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        vec_a = np.array(a, dtype=np.float32)
        vec_b = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))