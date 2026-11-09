Agent prompt: stop after two differently worded searches find nothing relevant, instead of searching repeatedly or reading whole documents to prove a negative.

Why: in v1 every unanswerable question took 7 tool calls and ~12.8K tokens (about 19x plain RAG) before the agent declined.

Command: python evaluation/run_answer_eval.py --variant v2 --app chat --concurrency 1

Result (10 November 2026): reverted. Answerable questions unchanged (26/26
after a parser fix for "[ file#chunk ]" with inner spaces). On the one
unanswerable question that ran, the agent ignored the rule: 7 tool calls,
14,173 tokens (v1: ~12,900), and it hit the 6-round cap, answering "I
couldn't finish looking this up" instead of a clean decline. The other 4
unanswerable questions hit Groq's free-tier tokens-per-day limit and were
not scored. Conclusion: a prompt-only rule doesn't change gpt-oss's search
behaviour; try a code-level limit next.
