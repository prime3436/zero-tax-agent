# ZERO-TAX Benchmark Report

- **Seed**: `42` | **Normal Tickets**: `200` | **Drift Wave**: `50` | **Teacher**: `simulated`

| Metric | Naive All-LLM Baseline | Zero-Tax LangGraph Engine | Improvement |
|---|---|---|---|
| Full-Graph Cost (INR) | Rs. 98.2969 | Rs. 55.9409 | **43.1% saved** |
| Distillable Nodes Cost (INR) | Rs. 59.6456 | Rs. 17.2895 | **71.0% saved** |
| Avg Latency / Ticket | 1839.7 ms | 995.0 ms | **45.9% faster** |
| Intent Accuracy | - | 100.0% | - |
| Policy Accuracy | - | 100.0% | - |
| Student Evaporation Rate | 0.0% | 71.0% | - |
| Low-Confidence Fallback Rate | - | 4.00% | - |
| Drift Detected Ticket Index | - | `201` | Auto-Demoted |
| Drift Recovery Ticket Index | - | `211` | Re-Promoted |
| Post-Recovery Accuracy | - | 100.0% | - |
