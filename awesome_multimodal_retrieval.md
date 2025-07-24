

# ColPali
- Thu, 27 Jun 2024 [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)
  - Better VLMs lead to better visual retrievers
    - Built on PaliGemma-3B -> ColPali
    - Built on Qwen2-VL 2B -> ColQwen2-VL
  - Visual Document Retrieval Benchmark ViDoRe
  - Our method, ColPali, significantly outperforms all other retrieval systems on ViDoRe while being fast and end-to-end trainable. 
  - These results demonstrate the potential and the many benefits of this novel Retrieval in Vision Space concept, 
  - which could significantly alter the way document retrieval is approached in the industry moving forward. 
- Wed, 2 Apr 2025 [Nomic Embed Multimodal: Open Source Multimodal Embedding Models for Text, Images, PDFs, and Charts](https://www.nomic.ai/blog/posts/nomic-embed-multimodal)
  - late 方法到底有没有用？
    - ColNomic Embed Multimodal 7B 62.7
    - ColNomic Embed Multimodal 3B 61.2
    - Nomic Embed Multimodal 7B 59.7
    - GME Qwen2 7B 59.0
    - Nomic Embed Multimodal 3B 58.8
- Mon, 23 Jun 2025 [jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval](https://arxiv.org/abs/2506.18902)
  - 对比 jina-embeddings-v4 (dense) 和 jina-embeddings-v4 (late)
  - J-VDR 73.98 vs 80.55， ViDoRe 84.11 vs 90.17， 好像 late 方法提升并不显著

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
- Mon, 2 Dec 2024 [LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](https://arxiv.org/abs/2412.01720)
  - Built on Qwen2-VL 7B
  - Training for Retrieval
    - stage-I: Adapting LMMs for Retrieval Tasks 
      - we start by adapting LMMs for text-to-text retrieval by training LoRA modules on the Natural Language Inference (NLI) dataset
      - 参考 E5-V 这确实是个高效的做法
    - Stage-II: Instruction Tuning for Universal Retrieval.
  - Training for Reranking
    - Collecting Training Data for Reranking
      - train the reranking model on its top 100 retrieved candidates and use them as hard negatives
    - Joint Training for Pointwise and Listwise Reranking.
      - Pointwise YES & NO
      - Listwise Llist = Lce(GT-POSITION, Reranker(q, cpos, c1, c2, · · · , cM)).
  - Ablation Study
    - Effectiveness of Two-stage Training
    - Scaling Trends of LamRA
    - Discussion on Pointwise and Listwise Rerank Methods. 好像Listwise效果略微好一点点
- Sun, 22 Dec 2024 [GME: Improving Universal Multimodal Retrieval by Multimodal LLMs](https://arxiv.org/abs/2412.16855)
  - GME-Qwen2-VL-2B & GME-Qwen2-VL-7B Built on Qwen2-VL 2B & Qwen2-VL 7B
  - Fused-Modal Data Synthesis
    - our preliminary experiments demonstrate that more diverse multimodal training data can further unlock the potential of MLLMs
    - motivates us to develop a training data synthesis pipeline and construct a large-scale
  - Training
    - Contrastive Learning， InfoNCE， Hard Negatives
    - 估计训练不足
  - Ablation Study on Modeling
    - Fine-tuning strategy, LoRA r=8 效果不错
    - Training data organization,  removal of hard negatives led to performance declines
    - Retrieval instructions, retrieval instructions are crucial for better UMR
    - Model Design
      - casual attention mode: negatively impact performance
      - use the EOS token state as the embedding: negatively impact performance
- Wed, 2 Apr 2025 [Nomic Embed Multimodal: Open Source Multimodal Embedding Models for Text, Images, PDFs, and Charts](https://www.nomic.ai/blog/posts/nomic-embed-multimodal)
  - Built on Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
  - This release includes four models, available in two sizes (3B and 7B parameters) and two variants:
    - ColNomic Embed Multimodal (3B and 7B): Multi-vector late interaction multimodal embedding models (more powerful)
    - Nomic Embed Multimodal (3B and 7B): Single-vector multimodal embedding models (faster & use less memory/storage)
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
  - dense vs late
    - 对比 jina-embeddings-v4 (dense) 和 jina-embeddings-v4 (late)
    - J-VDR 73.98 vs 80.55， ViDoRe 84.11 vs 90.17， 好像 late 方法提升并不显著
- Mon, 7 Jul 2025 [VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents](https://arxiv.org/abs/2507.04590)
  - Built on Qwen2-VL
  - MMEB-V2
    - Video Retrieval (V-RET)
    - Moment Retrieval (M-RET)
    - Video Classification (V-CLS)
    - Video QA (V-QA)
    - Visual Document Retrieval (VisDoc)
  - Main Results
    - 加入 VisDoc 数据，VisDoc Overall 分数相比 VLM2Vec-Qwen2VL 立竿见影，但是打不过 ColPali v1.3 (3B)
    - 等一下，GME (2B) & GME (7B) 没有使用 ColPali 为什么效果这么好
    - In visual document retrieval, VLM2Vec-V2 outperforms all VLM2Vec variants, 
    - although still trailing behind ColPali, which is specifically optimized for VisDoc tasks.

# Multilingual Multimodal Reranker
- Thu, 27 Mar 2025 [jinaai/jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0)
  - Base model: Qwen2-VL-2B-Instruct, utilizing its vision encoder, projection layer, and language model
  - Adaptation: Fine-tuned the language model with LoRA (Low-Rank Adaptation) techniques
  - Output layer: Post-trained MLP head to generate ranking scores measuring query-document relevance
  - Training objective: Optimized with pairwise and listwise ranking losses to produce discriminative relevance scores