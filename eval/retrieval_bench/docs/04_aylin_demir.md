# Aylin Demir (fictional candidate) — project notes

## Turkish natural language inference benchmark
- Created a Turkish NLI test set of 3,000 premise–hypothesis pairs written by native speakers, as an alternative to machine-translated XNLI.
- Showed that models score 9 points higher on translated XNLI than on native data, a sign of translation artifacts.
- Checked every item with back-translation to English and a second native annotator; disagreements were adjudicated by a third person.
- Evaluated mBERT, XLM-R and three instruction-tuned LLMs; reported accuracy per linguistic phenomenon such as negation and agglutinative suffixes.

## Turkish speech recognition
- Fine-tuned Whisper-small on 120 hours of Turkish podcast audio; word error rate dropped from 24.1% to 15.6%.
- Normalised numbers and suffixes before scoring, because Turkish word boundaries make WER unstable.

## Turkish sentiment dataset and annotation guidelines
- Collected 15,000 product reviews in Turkish and wrote a 12-page annotation guideline covering sarcasm, mixed sentiment and dialect spellings.
- Three annotators labelled each review; inter-annotator agreement was Krippendorff's alpha 0.74 after two rounds of guideline revisions.
- Released the dataset with a data card describing sources, licence, known biases and intended use.

## Teaching assistant: Introduction to NLP
- Prepared weekly labs on tokenization, n-gram language models and word embeddings for 80 students; graded assignments with an automatic test harness.

## Skills
- Python, PyTorch, Hugging Face Transformers, torchaudio; Turkish (native), English, German.
