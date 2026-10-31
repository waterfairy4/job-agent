# Judge evaluation

**55 items labeled.** Agreement 85%, Cohen's kappa 0.78.

- Judge more generous than you: 11%
- Judge stricter than you: 4%

## Confusion matrix (rows = you, columns = judge)

| you \ judge | strong | partial | missing |
| --- | --- | --- | --- |
| strong | 17 | 1 | 0 |
| partial | 1 | 17 | 1 |
| missing | 0 | 5 | 13 |

## By item type

| Type | Items | Agreement |
| --- | --- | --- |
| duty | 17 | 71% |
| must | 23 | 91% |
| nice | 15 | 93% |

## Match score: judge vs you (fully labeled jobs)

| Job | Judge score | Your score |
| --- | --- | --- |
| Huawei – AI Application Engineer - AI技术应用 | 56% | 56% |
| Huawei – AI Model Engineer – AI Model Evaluation | 71% | 71% |
| Unknown company – AI Large Model Evaluation Engineer | 74% | 65% |
| Huawei – Large Model Training and Inference Systems Researcher | 37% | 30% |

## Disagreements (for error analysis)

| Job | Item | You | Judge | Judge's reason | Your note |
| --- | --- | --- | --- | --- | --- |
| Huawei – AI Application Engineer - AI技术应用 | Conduct requirements analysis for LLM/AI algorithm applications | missing | partial | The evidence shows extensive LLM/AI work — evaluation pipelines, prompt/fine-tuning experiments, agent workflows, and multi-model benchmarking — but never describes requirements analysis (gathering or specifying application requirements) for LLM/AI applications. |  |
| Huawei – AI Application Engineer - AI技术应用 | Build AI data analysis, effect evaluation, issue diagnosis and continuous optimization to improve UX and business value | strong | partial | The evidence shows AI data analysis and effect evaluation work (end-to-end evaluation pipelines, statistical analysis, LoRA fine-tuning with measured alignment gains, agent-based evaluation flows), but nothing demonstrates issue diagnosis or continuous optimization aimed at improving UX and business value. |  |
| Unknown company – AI Large Model Evaluation Engineer | Teamwork and sense of responsibility | missing | partial | The evidence shows an internship in a hybrid workplace and independent/self-driven projects, which loosely suggests collaboration and self-management, but no passage explicitly documents teamwork or a sense of responsibility. |  |
| Unknown company – AI Large Model Evaluation Engineer | Relevant internship experience | partial | strong | The CV lists a software development internship at Netaş (July–September 2023) where the candidate built and deployed a full-stack web app with React, Java, Spring Boot, and Tailwind CSS. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Conduct innovative research on key technologies and system architectures for large model training and inference (Agent, Language, Multimodal, Generative Recommendation, Spatial Intelligence) | missing | partial | The evidence shows hands-on LLM work — LoRA fine-tuning of Qwen2.5-7B, multi-agent pipelines, evaluation infrastructure, and stated "large model system design" — but nothing on training/inference system architectures at scale, nor on multimodal, generative recommendation, or spatial intelligence. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Lead design and research of training and inference frameworks and acceleration features (pre-training, RL post-training, Agentic AI, inference engines, service-oriented architecture) | missing | partial | The evidence shows hands-on work on inference/evaluation pipelines, LoRA fine-tuning, multi-agent (agentic) flows, and REST API development, but not leadership of designing training and inference frameworks or acceleration features. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Track latest research and technology trends in AI cluster training/inference algorithms and system optimization | partial | missing | The evidence shows LLM evaluation, LoRA fine-tuning, multi-agent pipelines and general ML engineering, but nothing about tracking research/technology trends in AI cluster training or inference algorithms and system optimization. |  |
| Huawei – Large Model Training and Inference Systems Researcher | Proficiency in cutting-edge methods and tools for optimizing training and inference systems | missing | partial | The evidence shows hands-on LoRA fine-tuning, batch inference pipelines with caching, and evaluation tooling (PyTorch, PEFT/TRL, Hugging Face), but nothing on cutting-edge optimization methods for training/inference efficiency such as distributed training, quantization, or serving optimization. |  |
