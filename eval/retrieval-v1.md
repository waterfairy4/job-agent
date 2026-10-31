# Retrieval hit rate (检索命中率)

28 queries over 15 profile chunks, top 5.
**Warning:** with only 15 chunks, the top 5 holds a large share of the whole profile, so these numbers say little. Add more material to profile/ first.

| Retriever | Recall@5 | Hit@5 | MRR |
| --- | --- | --- | --- |
| bm25 | 83.3% | 92.9% | 0.929 |
| dense | 74.3% | 100.0% | 0.940 |
| hybrid | 88.6% | 100.0% | 1.000 |

Recall@5: share of the relevant chunks found in the top 5. Hit@5: share of queries with at least one relevant chunk in the top 5. MRR: 1 / rank of the first relevant chunk, averaged.
