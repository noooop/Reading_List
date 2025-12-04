
## memory management
- [Jenga: Effective Memory Management for Serving LLM with Heterogeneity](https://arxiv.org/abs/2503.18292)
  - First, recent models often have heterogeneous embeddings with different sizes:
    - VLMs
    - Mamba
  - Second, to handle long contexts more efficiently, some new LLM architectures use only a subset of the prefix tokens to generate the next token. 
    - sliding window