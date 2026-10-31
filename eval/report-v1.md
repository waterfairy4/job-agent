# Judge evaluation

**55 items labeled.** Agreement 75%, Cohen's kappa 0.62.

- Judge more generous than you: 16%
- Judge stricter than you: 9%

## Confusion matrix (rows = you, columns = judge)

| you \ judge | strong | partial | missing |
| --- | --- | --- | --- |
| strong | 14 | 4 | 0 |
| partial | 2 | 16 | 1 |
| missing | 0 | 7 | 11 |

## By item type

| Type | Items | Agreement |
| --- | --- | --- |
| duty | 17 | 59% |
| must | 23 | 78% |
| nice | 15 | 87% |

## Match score: judge vs you (fully labeled jobs)

| Job | Judge score | Your score |
| --- | --- | --- |
| Huawei – AI Application Engineer - AI技术应用 | 56% | 56% |
| Huawei – AI Model Engineer – AI Model Evaluation | 66% | 71% |
| Unknown company – AI Large Model Evaluation Engineer | 76% | 65% |
| Huawei – Large Model Training and Inference Systems Researcher | 37% | 30% |

## Disagreements (for error analysis)

| Job | Item | You | Judge | Judge's reason | Your note |
| --- | --- | --- | --- | --- | --- |
| Huawei – AI Application Engineer - AI技术应用 | Conduct requirements analysis for LLM/AI algorithm applications | missing | partial | The candidate has substantial LLM evaluation, fine-tuning, and agent-pipeline work (multilingual prompt construction, batch inference, statistical analysis, LoRA fine-tuning, agent workflows), but no evidence of conducting requirements analysis for LLM/AI algorithm applications. |  |
| Huawei – AI Application Engineer - AI技术应用 | Build AI data analysis, effect evaluation, issue diagnosis and continuous optimization to improve UX and business value | strong | partial | The candidate shows strong evaluation and analysis work (end-to-end evaluation pipelines, statistical analysis, model behavior experiments, agent-based validation), but there is no evidence of UX-focused analysis, issue diagnosis for product improvement, or continuous optimization tied to business value. |  |
| Huawei – AI Model Engineer – AI Model Evaluation | Design and develop evaluation sets, observation metrics and scoring standards | strong | partial | The candidate built end-to-end evaluation pipelines with multilingual prompt construction and statistical analysis, and ran large-scale model evaluations, but the evidence shows designing evaluation procedures rather than explicitly designing observation metrics and scoring standards. |  |
| Huawei – AI Model Engineer – AI Model Evaluation | Develop automated labeling and automatic clustering algorithms for automated analysis | missing | partial | The candidate shows automated evaluation pipelines, agentic workflows, and clustering-adjacent group/population modeling (seed-group framework, population-alignment fine-tuning), but no explicit automated labeling or automatic clustering algorithm development. |  |
| Huawei – AI Model Engineer – AI Model Evaluation | Familiar with deep learning frameworks | strong | partial | The candidate shows hands-on work with specific models (LoRA fine-tuning of Qwen2.5-7B, querying six instruction-tuned models, deep reinforcement learning coursework), which implies framework use, but no deep learning framework (e.g., PyTorch, TensorFlow) is explicitly named. |  |
| Unknown company – AI Large Model Evaluation Engineer | Familiar with deep learning frameworks | strong | partial | The CV shows hands-on work with large models (LoRA fine-tuning of Qwen2.5-7B, deep reinforcement learning coursework, LLM application engineering), which implies framework use but never names a specific deep learning framework such as PyTorch or TensorFlow. |  |
| Unknown company – AI Large Model Evaluation Engineer | Strong learning ability and self-motivation | partial | strong | The candidate independently designed and executed multiple self-directed research projects (LoRA fine-tuning, multi-agent evaluation pipeline, RL summarization), built an end-to-end evaluation pipeline from scratch, and maintains high GPAs (3.66/4.0, 3.6/4.0), demonstrating strong self-motivation and learning ability. |  |
| Unknown company – AI Large Model Evaluation Engineer | Teamwork and sense of responsibility | missing | partial | The CV shows internship experience and a self-driven first-author research project with an end-to-end pipeline, which hints at collaboration and responsibility, but no evidence explicitly documents teamwork or responsibility (e.g., team roles, leadership, or ownership statements). |  |
| Unknown company – AI Large Model Evaluation Engineer | Papers at top conferences such as NeurIPS, ICML, AAAI, VLDB | missing | partial | The CV mentions an AAAI 2026 published paper (referenced as a contrasting result) and a first-author paper under submission, but no evidence of the candidate's own paper accepted at NeurIPS, ICML, AAAI, or VLDB. |  |
| Unknown company – AI Large Model Evaluation Engineer | Relevant internship experience | partial | strong | The CV lists a software development internship at Netaş (July–September 2023) with concrete full-stack development responsibilities. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Conduct innovative research on key technologies and system architectures for large model training and inference (Agent, Language, Multimodal, Generative Recommendation, Spatial Intelligence) | missing | partial | The candidate shows strong LLM evaluation, LoRA fine-tuning, and multi-agent pipeline work, but there is no evidence of research on large-model training/inference system architectures or on multimodal, generative recommendation, or spatial intelligence. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Lead design and research of training and inference frameworks and acceleration features (pre-training, RL post-training, Agentic AI, inference engines, service-oriented architecture) | missing | partial | The candidate shows hands-on work in inference/evaluation pipelines, agentic multi-agent systems, LoRA fine-tuning, and RL-based training, but no evidence of leading design of training/inference frameworks, acceleration features, or service-oriented architecture. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Track latest research and technology trends in AI cluster training/inference algorithms and system optimization | partial | missing | The evidence shows LLM evaluation, LoRA fine-tuning, agent pipelines, and multilingual benchmarking, but nothing about tracking AI cluster training/inference algorithm or system optimization trends. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Proficiency in cutting-edge methods and tools for optimizing training and inference systems | missing | partial | The candidate shows LLM evaluation, batch inference pipelines, LoRA fine-tuning, and tooling like PyTorch/Hugging Face/PEFT/TRL, but there is no explicit evidence of optimizing training or inference system performance (e.g., efficiency, throughput, distributed training, quantization). |  |
