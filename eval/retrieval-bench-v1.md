# Retrieval hit rate, synthetic benchmark (检索命中率)

43 queries over 34 profile chunks, top 5.

| Retriever | Recall@5 | Hit@5 | MRR |
| --- | --- | --- | --- |
| bm25 | 74.4% | 79.1% | 0.773 |
| dense | 93.0% | 97.7% | 0.868 |
| hybrid | 81.4% | 83.7% | 0.778 |

Recall@5: share of the relevant chunks found in the top 5. Hit@5: share of queries with at least one relevant chunk in the top 5. MRR: 1 / rank of the first relevant chunk, averaged.
