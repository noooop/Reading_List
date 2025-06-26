

# ColPali
- Thu, 27 Jun 2024 [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
  - Visual Document Retrieval Benchmark ViDoRe
  - Our method, ColPali, significantly outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable. 
  - These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, 
  - which could significantly alter the way document retrieval is approached in the industry moving forward. 


# Retrieval(Embeddings) model
- Mon, 23 Jun 2025[jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval](https://arxiv.org/abs/2506.18902)
  - Built on Qwen/Qwen2.5-VL-3B-Instruct, jina-embeddings-v4 features:
  - features:
    - Matryoshka Representational Learning (MRL) as a way to train models for truncatable embedding vectors. 
    - The model incorporates task-specific Low-Rank Adaptation(LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-document retrieval, semantic text similarity, and code search.
    - ColPali: train a late-interaction embedding model to search document screenshots using text queries, performing significantly better than traditional approaches involving OCR and CLIPstyle models trained on image captions. 
  - evaluate & benchmark
    - Multilingual Text Retrieval：MTEB & MMTEB
    - Textual Semantic Similarity
    - LongEmbed benchmark
    - Multimodal Retrieval：Jina VDR， ViDoRe
    - Code Retrieval：MTEB-CoIR