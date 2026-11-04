# Retrieval hit rate (检索命中率)

28 queries over 18 profile chunks, top 5.

| Retriever | Recall@5 | Hit@5 | MRR |
| --- | --- | --- | --- |
| bm25 | 81.3% | 92.9% | 0.911 |
| dense | 72.6% | 100.0% | 0.940 |
| hybrid | 85.3% | 100.0% | 0.946 |
| rerank | 87.8% | 100.0% | 0.976 |

Recall@5: share of the relevant chunks found in the top 5. Hit@5: share of queries with at least one relevant chunk in the top 5. MRR: 1 / rank of the first relevant chunk, averaged.

Settings: 18 chunks, chunk size 600, BAAI/bge-m3, section chunks off, BM25 weight 1.0, reranker BAAI/bge-reranker-v2-m3 top 20.
