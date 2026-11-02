# Retrieval hit rate (检索命中率)

28 queries over 28 profile chunks, top 5.

| Retriever | Recall@5 | Hit@5 | MRR |
| --- | --- | --- | --- |
| bm25 | 74.9% | 92.9% | 0.848 |
| dense | 73.2% | 100.0% | 0.940 |
| hybrid | 82.3% | 100.0% | 0.964 |

Recall@5: share of the relevant chunks found in the top 5. Hit@5: share of queries with at least one relevant chunk in the top 5. MRR: 1 / rank of the first relevant chunk, averaged.
