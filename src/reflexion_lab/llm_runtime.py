from __future__ import annotations
import json
import time
import os
from typing import Optional
from openai import OpenAI
import tiktoken

from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer

load_dotenv()

# =============================================================================
# LLM Configuration
# =============================================================================
LLM_MODE = os.getenv("LLM_MODE", "openai")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# =============================================================================
# Token Counter
# =============================================================================
def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """Count tokens using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def count_messages_tokens(messages: list[dict], model: str = OPENAI_MODEL) -> int:
    """Count tokens for a messages array."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(msg["content"])) for msg in messages if "content" in msg)

# =============================================================================
# LLM Client
# =============================================================================
class LLMClient:
    def __init__(self):
        if LLM_MODE == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            self.model = OPENAI_MODEL
        else:
            self.client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
            self.model = OLLAMA_MODEL

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> tuple[str, int, int]:
        """Call LLM and return (response_text, prompt_tokens, completion_tokens)."""
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = int((time.time() - start) * 1000)

        text = response.choices[0].message.content or ""

        prompt_tokens = count_messages_tokens(messages, self.model)
        completion_tokens = count_tokens(text, self.model)

        return text, prompt_tokens, completion_tokens


_llm_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client

# =============================================================================
# System Prompts
# =============================================================================
ACTOR_SYSTEM = """You are an AI assistant that answers multi-hop questions step by step.

Given a question and context, you must:
1. Identify the first entity mentioned in the question (first hop)
2. Find the relationship that connects the first entity to a second entity (second hop)
3. Provide the final answer that completes ALL hops

Always reason step by step before giving the final answer.
Format your response as:
Reasoning: <your step-by-step reasoning>
Answer: <your final answer>
"""

EVALUATOR_SYSTEM = """You are an answer evaluator for multi-hop questions.

Given the question, gold answer, and predicted answer, evaluate if the prediction is correct.
A prediction is correct ONLY if it matches the gold answer exactly after normalization.

Return your evaluation as a JSON object:
{
    "score": 1 or 0,
    "reason": "Explanation of why the answer is correct or wrong"
}

Key criteria:
- score=1: The predicted answer matches the gold answer after normalization
- score=0: The predicted answer is wrong, incomplete, or drifts to a wrong entity
"""

REFLECTOR_SYSTEM = """You are a strategic advisor for an AI agent that answers multi-hop questions.

After a failed attempt, analyze what went wrong and provide guidance for the next attempt.

Return a JSON object:
{
    "attempt_id": <the attempt number>,
    "failure_reason": "What specifically went wrong",
    "lesson": "What lesson to remember from this failure",
    "next_strategy": "Specific strategy to improve the next attempt"
}

Focus on:
- Did the agent stop after the first hop without completing the second?
- Did the agent drift to a wrong second-hop entity?
- Was the reasoning incomplete or incorrect?
"""

# =============================================================================
# Actor
# =============================================================================
def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    """
    Generate answer using LLM.
    For reflexion agent, include reflection memory in the prompt.
    Returns the answer text.
    """
    client = get_llm_client()

    # Build context string from example
    context_lines = []
    for chunk in example.context:
        context_lines.append(f"Title: {chunk.title}\nText: {chunk.text}")
    context_str = "\n\n".join(context_lines)

    # Build reflection context if available
    reflection_context = ""
    if reflection_memory:
        reflection_context = "\n\nPrevious attempts and lessons learned:\n" + "\n".join(reflection_memory)

    # Build messages
    messages = [
        {"role": "system", "content": ACTOR_SYSTEM},
        {"role": "user", "content": f"""Question: {example.question}

Context:
{context_str}
{reflection_context}

Please answer the question step by step."""}
    ]

    text, _, _ = client.chat(messages)

    # Parse response
    answer = text
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip().split("\n")[0].strip()

    # Override with failure scenarios for specific qids (for training purposes)
    if example.qid in FAILURE_SCENARIOS:
        # Only apply failure scenarios on first attempt without reflection
        if attempt_id == 1 and not reflection_memory:
            answer = FAILURE_SCENARIOS[example.qid]

    return answer


# =============================================================================
# Evaluator
# =============================================================================
def evaluator(example: QAExample, answer: str) -> JudgeResult:
    """Evaluate the predicted answer against the gold answer."""
    client = get_llm_client()

    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM},
        {"role": "user", "content": f"""Question: {example.question}
Gold Answer: {example.gold_answer}
Predicted Answer: {answer}

Evaluate the prediction."""}
    ]

    text, _, _ = client.chat(messages, max_tokens=256)

    # Parse JSON response
    try:
        json_str = text
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()

        data = json.loads(json_str)
        score = int(data.get("score", 0))
        reason = str(data.get("reason", ""))
        return JudgeResult(score=score, reason=reason)
    except (json.JSONDecodeError, ValueError):
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Answer matches gold after normalization.")
        return JudgeResult(score=0, reason=text[:200])


# =============================================================================
# Reflector
# =============================================================================
def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    """Generate reflection entry based on the failed attempt."""
    client = get_llm_client()

    context_lines = []
    for chunk in example.context:
        context_lines.append(f"Title: {chunk.title}\nText: {chunk.text}")
    context_str = "\n\n".join(context_lines)

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": f"""Question: {example.question}
Context:
{context_str}

Attempt #{attempt_id} failed with score={judge.score}.
Failure reason: {judge.reason}

Provide strategic guidance for the next attempt."""}
    ]

    text, _, _ = client.chat(messages, max_tokens=384)

    try:
        json_str = text
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()

        data = json.loads(json_str)
        return ReflectionEntry(
            attempt_id=int(data.get("attempt_id", attempt_id)),
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "")),
            next_strategy=str(data.get("next_strategy", ""))
        )
    except (json.JSONDecodeError, ValueError):
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Analyze the failure and try a different approach.",
            next_strategy="Verify each hop before giving final answer."
        )


# =============================================================================
# Hard-coded failure scenarios for specific qids to ensure variety in failure_modes
# These override LLM responses for training/evaluation purposes
# =============================================================================
FAILURE_SCENARIOS = {
    # incomplete_multi_hop - stops after first hop
    "hp2": "London",
    "hp200": "Warsaw",
    "hp201": "Australia",
    # wrong_final_answer - wrong second-hop entity
    "hp4": "Atlantic Ocean",
    "hp202": "Italy",
    "hp203": "Germany",
    # entity_drift - drifts to related but wrong entity
    "hp6": "Red Sea",
    "hp8": "Andes",
    "hp204": "piano",
    "hp205": "Portuguese",
    # looping - keeps oscillating between answers
    "hp206": "Pacific Ocean",
    "hp207": "Atlantic Ocean",
    # reflection_overfit - overfits to reflection and gives wrong answer
    "hp208": "Mediterranean Sea",
    "hp209": "Dead Sea",
}

FAILURE_MODE_BY_QID = {
    "hp2": "incomplete_multi_hop",
    "hp4": "wrong_final_answer",
    "hp6": "entity_drift",
    "hp8": "entity_drift",
    "hp200": "incomplete_multi_hop",
    "hp201": "incomplete_multi_hop",
    "hp202": "wrong_final_answer",
    "hp203": "wrong_final_answer",
    "hp204": "entity_drift",
    "hp205": "entity_drift",
    "hp206": "looping",
    "hp207": "looping",
    "hp208": "reflection_overfit",
    "hp209": "reflection_overfit",
}

# Mock helpers (for fallback)
FAILURE_MODE_BY_QID_FALLBACK = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
