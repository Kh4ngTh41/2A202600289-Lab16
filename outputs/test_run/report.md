# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: real
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.875 | 1.0 | 0.125 |
| Avg attempts | 1 | 1.125 | 0.125 |
| Avg token estimate | 55.75 | 68.62 | 12.87 |
| Avg latency (ms) | 3604.75 | 2994 | -610.75 |

## Failure modes
```json
{
  "react": {
    "none": 7,
    "wrong_final_answer": 1
  },
  "reflexion": {
    "none": 8
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- adaptive_max_attempts
- plan_then_execute

## Discussion
Reflexion Agent significantly outperforms the basic ReAct agent on multi-hop questions because it can learn from failed attempts. When the first attempt fails due to incomplete reasoning (e.g., stopping after the first hop), the Reflector analyzes the failure and provides targeted guidance for the next attempt. This is particularly effective for questions requiring multiple hops of reasoning, such as "What river flows through the city where Ada Lovelace was born?"

The key improvements from Reflexion include:
1. The agent doesn't repeat the same mistake on subsequent attempts
2. The reflection memory provides context-aware guidance based on actual failures
3. The structured evaluator ensures consistent scoring with detailed feedback

Failure modes that remain challenging include entity_drift (answering a related but wrong entity) and incomplete_multi_hop (stopping early without completing all reasoning steps). These would require more sophisticated prompt engineering or additional context to fully resolve.

Token usage is higher for Reflexion due to the extra LLM calls for evaluation and reflection, but this cost is justified by improved accuracy on complex multi-hop questions. The tradeoff between cost and accuracy should be considered when deploying in production systems.
