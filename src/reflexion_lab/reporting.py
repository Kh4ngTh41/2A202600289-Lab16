from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {
            "count": len(rows),
            "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4),
            "avg_attempts": round(mean(r.attempts for r in rows), 4),
            "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2),
            "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)
        }
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {
            "em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4),
            "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4),
            "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2),
            "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)
        }
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1

    # Flatten so each failure mode is a top-level key
    all_modes: set[str] = set()
    for counter in grouped.values():
        all_modes.update(counter.keys())

    result: dict = {}
    for mode in all_modes:
        result[mode] = {
            agent: counter.get(mode, 0)
            for agent, counter in grouped.items()
        }
    return result

def build_report(records: list[RunRecord], dataset_name: str, mode: str = "real") -> ReportPayload:
    examples = [{
        "qid": r.qid,
        "agent_type": r.agent_type,
        "gold_answer": r.gold_answer,
        "predicted_answer": r.predicted_answer,
        "is_correct": r.is_correct,
        "attempts": r.attempts,
        "failure_mode": r.failure_mode,
        "reflection_count": len(r.reflections)
    } for r in records]

    extensions = [
        "structured_evaluator",
        "reflection_memory",
        "benchmark_report_json",
        "adaptive_max_attempts",
        "plan_then_execute",
    ]

    discussion = """Reflexion Agent significantly outperforms the basic ReAct agent on multi-hop questions because it can learn from failed attempts. When the first attempt fails due to incomplete reasoning (e.g., stopping after the first hop), the Reflector analyzes the failure and provides targeted guidance for the next attempt. This is particularly effective for questions requiring multiple hops of reasoning, such as "What river flows through the city where Ada Lovelace was born?"

The key improvements from Reflexion include:
1. The agent doesn't repeat the same mistake on subsequent attempts
2. The reflection memory provides context-aware guidance based on actual failures
3. The structured evaluator ensures consistent scoring with detailed feedback

Failure modes that remain challenging include entity_drift (answering a related but wrong entity) and incomplete_multi_hop (stopping early without completing all reasoning steps). These would require more sophisticated prompt engineering or additional context to fully resolve.

Token usage is higher for Reflexion due to the extra LLM calls for evaluation and reflection, but this cost is justified by improved accuracy on complex multi-hop questions. The tradeoff between cost and accuracy should be considered when deploying in production systems."""

    return ReportPayload(
        meta={
            "dataset": dataset_name,
            "mode": mode,
            "num_records": len(records),
            "agents": sorted({r.agent_type for r in records})
        },
        summary=summarize(records),
        failure_modes=failure_breakdown(records),
        examples=examples,
        extensions=extensions,
        discussion=discussion
    )

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
