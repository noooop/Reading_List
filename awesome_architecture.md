
# Bert like (encode only)
- Thu, 11 Oct 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Fri, 5 Jun 2020 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
  - mask language modeling (MLM)
  - Disentangled attention (the attention weight of a word pair can be computed as a sum of four attention scores
using disentangled matrices on their contents and positions as content-to-content, content-to-position,
position-to-content, and position-to-position)
- Thu, 18 Nov 2021 [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)
  - replaced token detection (RTD) (ELECTRA-Style Pre-Training)
- Mon, 30 Oct 2023 [Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents](https://arxiv.org/abs/2310.19923)
  - BERT with ALiBi, GEGLU, BF16, mean pooling
- Fri, 29 Dec 2023 [MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining](https://arxiv.org/abs/2312.17482)
  - This architecture combines FlashAttention [11], ALiBi [44], Gated Linear Units[12, 50], a dynamic unpadding module [66], and low precision LayerNorm.
- Fri, 2 Feb 2024 [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613)
  - NomicBertModel 架构比较现代 bert_with_rope
- Tue, 9 Apr 2024 [LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders](https://arxiv.org/abs/2404.05961)
  - additional training phase with a specially designed masked token prediction to warm-up the bidirectional attention.
- Mon, 29 Jul 2024 [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://arxiv.org/abs/2407.19669)
  - BERT + RoPE + GLU + xformers， 12 层 768 维，306M 比 bge m3 小，  [CLS] pooling
- Wed, 18 Dec 2024 [ModernBERT](https://arxiv.org/abs/2412.13663)
  - reduce Bias Terms, GeGLU, Rotary
  - Alternating Attention, 每 3 层部署全局注意力, 其余层则采用128 token 滑动窗口的局部注意力
  - Model Design, 深而窄的架构, 渐进式参数空间扩展
  - Training Settings: MLM use a masking rate of 30 percent, Warmup-Stable-Decay (WSD)
  - Weight Initialization and Tiling
  - 词表考虑和code数据加入
- Tue, 11 Feb 2025 [Training Sparse Mixture Of Experts Text Embedding Models](https://arxiv.org/abs/2502.07972)
  - Embedding Models 进入 Mixture Of Experts 时代 
- Tue, 22 Apr 2025 [腾讯Conan-Embedding-V2发布](https://zhuanlan.zhihu.com/p/1897675709696149020)
  - SoftMask (LLM2Vec)
- Thu, 5 Jun 2025 [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)
  - The Qwen3 embedding and reranking models are built on the dense version of Qwen3 foundation models and are available in three sizes: 0.6B, 4B, and 8B parameters

# Positional Encoding
- Mon, 12 Jun 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - In this work, we use sine and cosine functions of different frequencies
- Thu, 11 Oct 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Learnable Token Embeddings + Segment Embeddings + Position Embeddings
- Sun, 28 Jun 2020 [Rethinking Positional Encoding in Language Pre-training](https://arxiv.org/abs/2006.15595)
  - 位置嵌入应该在attention里面计算，而不是在加在词向量里
- Tue, 20 Apr 2021 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  - 大名鼎鼎的 Rotary Position Embedding

