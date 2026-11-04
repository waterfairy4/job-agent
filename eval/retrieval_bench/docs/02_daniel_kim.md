# Daniel Kim (fictional candidate) — project notes

## Legal document search assistant
- Built a retrieval-augmented assistant for a law firm over 40,000 contracts and court rulings.
- Hybrid retrieval: Elasticsearch BM25 plus OpenAI text-embedding-3-small vectors, merged with a weighted sum tuned on a validation set.
- A ColBERT late-interaction model re-scores the top 50 passages; this raised citation accuracy from 78% to 91% on 400 lawyer-written questions.
- Every answer quotes the clause it relies on; answers without a matching clause are refused.
- Scanned PDFs are converted with Tesseract OCR and a layout model that keeps table cells together.

## Contract clause extraction
- Fine-tuned a DeBERTa-v3 token classifier to extract termination, liability and payment clauses; F1 0.88 across 12 clause types.
- Active learning: lawyers labelled only the 1,500 most uncertain sentences, which halved annotation time.

## E-commerce search ranking (internship)
- Learning-to-rank for an online shop: a LambdaMART model (LightGBM) re-ranks the top 200 products from a keyword engine.
- Features include text match scores, price, click-through rate and freshness; labels come from clicks corrected for position bias.
- Offline NDCG@10 improved from 0.412 to 0.447; an online A/B test raised revenue per search by 2.3%.
- Built a query-understanding step that maps misspelled brand names to the right brand.

## Receipt OCR side project
- Extracted totals and dates from photographed receipts with PaddleOCR and simple layout rules; 89% exact match on 500 receipts.

## Skills
- Python, TypeScript; Elasticsearch, PyTorch, Hugging Face; AWS Lambda.
