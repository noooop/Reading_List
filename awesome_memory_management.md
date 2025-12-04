
## memory management
- Tue, 12 Sep 2023 [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
  - PagedAttention
- Mon, 24 Mar 2025 [Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://arxiv.org/abs/2503.18292)
  - First, recent models often have heterogeneous embeddings with different sizes:
    - VLMs
    - Mamba
  - Second, to handle long contexts more efficiently, some new LLM architectures use only a subset of the prefix tokens to generate the next token. 
    - sliding window