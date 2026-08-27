# Awesome LLMs 2026

### Qwen
- [Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
  - https://huggingface.co/collections/Qwen/qwen38-flash-next
  - https://qwen.ai/blog?id=qwen3.8-flash-next
  - Highlights
    - Hybrid Attention with QSA (Gated DeltaNet and Qwen Sparse Attention (QSA)) 1:3
    - Gated Residual
    - N-gram Embedding
  - Architecture
    - Language Model
      - Number of Parameters: 125B with 6B activated, plus 51B n-gram embedding and 4B MTP
      - Number of Layers: 48
      - Hidden Dimension: 2560
      - Hidden Layout: 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Qwen Sparse Attention → MoE))
      - Mixture Of Experts
        - Number of Experts: 512
        - Number of Activated Experts: 10 Routed + 1 Shared
        - Expert Intermediate Dimension: 640
    - Vision
      - 与 Qwen3-vl 相同