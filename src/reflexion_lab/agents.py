from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .llm_runtime import actor_answer, evaluator, reflector, get_llm_client, count_tokens, count_messages_tokens
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord, JudgeResult

FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        total_tokens = 0
        total_latency_ms = 0

        for attempt_id in range(1, self.max_attempts + 1):
            client = get_llm_client()

            # Build context for token counting
            context_lines = []
            for chunk in example.context:
                context_lines.append(f"Title: {chunk.title}\nText: {chunk.text}")
            context_str = "\n\n".join(context_lines)

            # Count prompt tokens before calling actor
            prompt_text = f"Question: {example.question}\n\nContext:\n{context_str}"
            if reflection_memory:
                prompt_text += "\n\nPrevious attempts and lessons learned:\n" + "\n".join(reflection_memory)
            prompt_tokens = count_messages_tokens([
                {"content": prompt_text}
            ], client.model)

            # Call actor
            start = __import__("time").time()
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            actor_latency = int((__import__("time").time() - start) * 1000)

            # Count completion tokens
            completion_tokens = count_tokens(answer, client.model)
            token_estimate = prompt_tokens + completion_tokens

            # Call evaluator
            start = __import__("time").time()
            judge = evaluator(example, answer)
            eval_latency = int((__import__("time").time() - start) * 1000)

            latency_ms = actor_latency + eval_latency
            total_latency_ms += latency_ms
            total_tokens += token_estimate

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            # Reflexion logic for reflexion agent
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)

                # Build reflection memory string for next attempt
                mem_entry = (
                    f"[Attempt {attempt_id}] "
                    f"Lesson: {reflection.lesson} | "
                    f"Strategy: {reflection.next_strategy}"
                )
                reflection_memory.append(mem_entry)
                trace.reflection = reflection

            traces.append(trace)

        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency_ms,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)


class AdaptiveReflexionAgent(BaseAgent):
    """Reflexion Agent with adaptive max_attempts based on question difficulty."""

    DIFFICULTY_MAX_ATTEMPTS = {
        "easy": 2,
        "medium": 3,
        "hard": 5,
    }

    def __init__(self) -> None:
        super().__init__(agent_type="reflexion", max_attempts=3)

    def run(self, example: QAExample) -> RunRecord:
        difficulty = example.difficulty if hasattr(example, "difficulty") else "medium"
        self.max_attempts = self.DIFFICULTY_MAX_ATTEMPTS.get(difficulty, 3)
        return super().run(example)
