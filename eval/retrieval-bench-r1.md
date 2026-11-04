# Retrieval hit rate, synthetic benchmark (检索命中率)

43 queries over 34 profile chunks, top 5.

| Retriever | Recall@5 | Hit@5 | MRR |
| --- | --- | --- | --- |
| bm25 | 74.4% | 79.1% | 0.773 |
| dense | 93.0% | 97.7% | 0.868 |
| hybrid | 81.4% | 83.7% | 0.778 |
| rerank | 96.5% | 97.7% | 0.977 |

Recall@5: share of the relevant chunks found in the top 5. Hit@5: share of queries with at least one relevant chunk in the top 5. MRR: 1 / rank of the first relevant chunk, averaged.

Settings: 34 chunks, chunk size 600, BAAI/bge-m3, section chunks off, BM25 weight 1.0, reranker BAAI/bge-reranker-v2-m3 top 20.
