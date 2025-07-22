

# ColPali
- Thu, 27 Jun 2024 [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
  - Visual Document Retrieval Benchmark ViDoRe
  - Our method, ColPali, significantly outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable. 
  - These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, 
  - which could significantly alter the way document retrieval is approached in the industry moving forward. 


# Multilingual Multimodal Retrieval(Embeddings)
## CLIP
- Mon, 30 Oct 2023 [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)
- Thu, 30 May 2024 [Jina CLIP: Your CLIP Model Is Also Your Text Retriever](https://arxiv.org/abs/2405.20204)
  - txt: BERT with AliBi(jina-embeddings-v2), img: EVA02 ViT B/16 (224*224) 
  - Our experiments show that EVA02 significantly outperforms comparable image encoders like DinoV2 (Oquab et al., 2024) and ViT B/16 models from OpenCLIP (Ilharco et al., 2021)
  - Training Recipe (three-stage training) 
    - Stage 1 focuses on learning to align image and text representations while minimizing losses in text-text performance. To this end, we train on large-scale and weakly
supervised text-image and text-text pair datasets.
    - Stage 2 presents longer, synthetic image captions to
the model while continuing to train with text-text pairs.
    - Stage 3 uses hard negatives to further improve the text
encoder in separating relevant from irrelevant text. To
maintain text-image alignment, we continue training
on long image captions.
- Mon, 16 Sep 2024 [jina-embeddings-v3: Multilingual Embeddings With Task LoRA](https://arxiv.org/abs/2409.10173)
- Wed, 11 Dec 2024 [jina-clip-v2: Multilingual Multimodal Embeddings for Text and Images](https://arxiv.org/abs/2412.08802)
  - txt: BERT with Rope(jina-embeddings-v3), img: EVA02 ViT L/14 (512*512) 
  - Training Recipe (three-stage training) 
    - Stage 1 image resolution (224, 224) → (384, 384)
    - Stage 2 image resolution (384, 384)
    - Stage 3 image resolution (512, 512)
  - ANALYSIS
    - THE ROLE OF IMAGE RESOLUTION IN VISUAL DOCUMENT RETRIEVAL
      - Unsurprisingly, increasing image resolution has a positive impact on linking queries to visually rich documents.
      - We consider (512, 512) the optimal image resolution for this model, with a reasonable balance between performance and cost.
## VLM2Vec
- Wed, 17 Jul 2024 [E5-V: Universal Embeddings with Multimodal Large Language Models](https://arxiv.org/abs/2407.12580)
  - Built on LLaVA-NeXT-8B(LLaVA-1.6)
  - modality gap
    - Compared to CLIP, although MLLM represents the image and text with the same encoder, 
    - the multimodal embeddings from MLLM show a clear modality gap between text and image embeddings.
  - Single Modality Training
    - E5-V trains MLLMs with contrastive learning on text pairs
    - 这也太省钱了
- Mon, 7 Oct 2024 [VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks](https://arxiv.org/abs/2410.05160)
  - Built on Phi-3.5-V LLaVA-1.6 (LLaVA-1.6比Phi-3.5-V好一些，尤其是使用res=1344×1344作为输入)
  - VLM2VEC:
    - (1) VLMs are trained on massive multimodal datasets and can handle any combination of images and text, as well as high-resolution images and long text inputs; 
    - (2) vision and language features are deeply fused in the transformer model, improving the model’s ability to capture cross-modal relationships; 
    - and (3) these models are well-suited for generalizing across diverse tasks, particularly those requiring instruction-following capabilities.
  - MMEB
    - We introduce a novel benchmark, MMEB (Massive Multimodal Embedding Benchmark), which includes 36 datasets spanning four meta-task categories
    - classification, visual question answering, retrieval, and visual grounding
- Mon, 23 Jun 2025 [jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval](https://arxiv.org/abs/2506.18902)
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

# Multilingual Multimodal Reranker
- Thu, 27 Mar 2025 [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)
  - Base model: Qwen2-VL-2B-Instruct, utilizing its vision encoder, projection layer, and language model
  - Adaptation: Fine-tuned the language model with LoRA (Low-Rank Adaptation) techniques
  - Output layer: Post-trained MLP head to generate ranking scores measuring query-document relevance
  - Training objective: Optimized with pairwise and listwise ranking losses to produce discriminative relevance scores