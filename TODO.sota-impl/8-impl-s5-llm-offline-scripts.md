# Impl S5: offline LLM loop scripts (graph node S5)

## Status

scripts-first. Deliverables: distill/audit/hard-negative scripts with cached-artifact outputs and bounded budgets. Execution requires an owner-provided LLM API key (no billing exists in this org — never assumed otherwise); S1 synthetic data substitutes for v1 training so S2/S3 are unblocked regardless.

## Gate

G-S5 (partial without key): scripts run end-to-end on a 100-pair sample once a key exists; until then, S1 substitutes.

## Consumers

S2/S3 data quality; S8 split audits.
