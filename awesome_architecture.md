
# Survey
- 19 Jul 2025 [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- 09 Aug 2025 [From GPT-2 to gpt-oss: Analyzing the Architectural Advances](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the)

# Bert like (encode only)
- Thu, 11 Oct 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Bidirectional： Such restrictions(unidirectional) are sub-optimal for sentence-level tasks,
and could be very harmful when applying finetuning based approaches to token-level tasks such
as question answering, where it is crucial to incorporate context from both directions
  - Model Architecture
    - BERTBASE (L=12, H=768, A=12, Total Parameters=110M) 
    - BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M).
  - Input/Output Representations
    - The first token of every sequence is always a special classification token ([CLS]).
    - The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. 
    - Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways.
      - First, we separate them with a special token ([SEP]). 
      - Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B.
    - For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings.
  - Pre-training BERT
    - Task #1: Masked LM
    - Task #2: Next Sentence Prediction (NSP)
- Fri, 5 Jun 2020 [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
  - mask language modeling (MLM)
  - Disentangled attention (the attention weight of a word pair can be computed as a sum of four attention scores
using disentangled matrices on their contents and positions as content-to-content, content-to-position,
position-to-content, and position-to-position)
- Fri, 26 Jul 2019 [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
  - we instead consider training BERT with a larger byte-level BPE vocabulary containing 50K subword units
  - removing the NSP loss matches or slightly improves downstream task performance
    - 这里就有一个小细节
      - Bert 的 tokenizer 会输出 token_type_ids， 比如 cross-encoder/ms-marco-TinyBERT-L-2-v2
      - RoBERTa 的 tokenizer 不会，比如 BAAI/bge-reranker-base
  - Static vs. Dynamic Masking
  - Training with large batches
  - we pretrain RoBERTa for significantly longer, increasing the number of pretraining steps from 100K to 300K, and then further to 500K
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
- Thu, 18 Apr 2024 [LongEmbed: Extending Embedding Models for Long Context Retrieval](https://arxiv.org/abs/2404.12096)
   - 检索模型进入长上下文时代，RoPE 含金量还在不断上升
   - LONGEMBED benchmark, which includes two synthetic and four real-world tasks
   - we pretrain E5-RoPE following the training procedure and data of E5.
   - 通过控制变量，对比 E5 和 E5-RoPE， 确认 RoPE 能显著提高长上下文能力
- Mon, 15 Jun 2024 [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) [gte-Qwen1.5-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct) 
  - gte-Qwen2-7B-instruct is the latest model in the gte (General Text Embedding) model family that ranks No.1 in both English and Chinese evaluations on the Massive Text Embedding Benchmark MTEB benchmark (as of June 16, 2024).
  - LLM as Retrieval + LLM2Vec
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
- Fri, 7 Mar 2025 [EuroBERT: Scaling Multilingual Encoders for European Languages](https://arxiv.org/abs/2503.05500)
  - Architecture
    The EuroBERT models are based on a standard dense transformer (Vaswani et al., 2017),
    with several architectural changes. Similarly to Llama 2 (Touvron et al., 2023), we remove
    all biases. Additionally, we incorporate grouped query attention (Ainslie et al., 2023), swish
    gated linear units (Shazeer, 2020), root mean square layer normalization (Zhang & Sennrich,
    2019), and rotary position embeddings (Su et al., 2024).
- Thu, 5 Jun 2025 [Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models](https://arxiv.org/abs/2506.05176)
  - 没使用  bidirectional attention 是不是逆潮流而动
  - The Qwen3 embedding and reranking models are built on the dense version of Qwen3 foundation models and are available in three sizes: 0.6B, 4B, and 8B parameters
- Tue, 1 Jul 2025 [Should We Still Pretrain Encoders with Masked Language Modeling?](https://arxiv.org/abs/2507.00994)
  - 这个实验和结论真的太扎实了 
  - Pretraining with CLM or MLM
    - MLM generally outperforms CLM on text representation tasks.
    - There is no universally optimal masking ratio.
    - CLM models can perform competitively
    - CLM is more data-efficient than MLM in the early stages of training.
    - CLM-based pretraining improves fine-tuning stability.
  - Two-Stage CLM+MLM Pretraining
    - Under fixed compute constraints, starting pretraining with CLM and continuing with MLM yields better results than MLM alone. 
      - Overall, a split between 25%-75% and 50%-50% seems to provide the best balance.
    - CLM-based models exhibit lower sensitivity to masking ratio.
  - Continued Pretraining from CLM and MLM Models
    - MLM CPT on a CLM-pretrained model outperforms MLM-only training.
    - Fewer CPT steps already show strong performance

# Positional Encoding
- Mon, 12 Jun 2017 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - In this work, we use sine and cosine functions of different frequencies
- Thu, 11 Oct 2018 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - Learnable Token Embeddings + Segment Embeddings + Position Embeddings
- Wed, 23 Oct 2019 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
  - the T5 model of Raffel et al. (2020) uses a relative position method (Shaw et al., 2018; Huang et al., 2019) 
  - that adds no position information to word embeddings (as in the previous method)
- Sun, 28 Jun 2020 [Rethinking Positional Encoding in Language Pre-training](https://arxiv.org/abs/2006.15595)
  - 位置嵌入应该在attention里面计算，而不是在加在词向量里
- Tue, 20 Apr 2021 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
  - 大名鼎鼎的 Rotary Position Embedding
- Fri, 27 Aug 2021 [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)
  - trained on shorter-L sequences and assumed to generalize to longer contexts at inference time
  - We find that transformer language models (LMs) that use sinusoidal position embeddings have very weak extrapolation abilities
  - We therefore introduce Attention with Linear Biases (ALiBi) to facilitate efficient extrapolation. 
  - ALiBi negatively biases attention scores with a linearly decreasing penalty proportional to the distance between the relevant key and query
  - ATTENTION WITH LINEAR BIASES (ALIBI)
    - When using ALiBi, we do not add position embeddings at any point in the network. The only modification we apply is after the query-key dot product, where we add a static, non-learned bias
    - softmax(qiK> + m · [−(i − 1), ..., −2, −1, 0])

# DeepSeek
- Fri, 27 Dec 2024 [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  - DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token
- Wed, 22 Jan 2025 [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
  - GRPO
- Mon, 29 Sep 2025 [DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/commits/main/DeepSeek_V3_2.pdf)
  - DeepSeek Sparse Attention (DSA)