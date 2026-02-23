
# Reasoning
- Mon, 16 Jun 2025 [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
  - architecture
    - 456B-A45.9B
    - MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism.
  - Computational Precision Mismatch in Generation and Training
    - Through layer-by-layer analysis, we identified high-magnitude activations in the LM head at the output layer as the primary source of error. 
    - To address this, we increased the precision of the LM output head to FP32
- Fri, 8 Aug 2025 [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471)
  - architecture
    - 355B-A32B & 106B-A12B
    - we reduce the width (hidden dimension and number of routed experts) of the model and increase its height (number of layers), 
    - as we found that deeper models exhibited better reasoning capacity.
  - Agentic, Reasoning, and Coding
- Tue, 17 Feb 2026 [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
  - architecture
    - 744B + 28.5T tokens
      - GLM-5 scales to 256 experts and reduces its layer count to 80 to minimize
expert parallelism communication overhead. This results in a 744B parameter model (40B active
parameters), doubling the total size of GLM-4.5, which utilized 355B total and 32B active parameters
    - MLA-256
      - we increase the head dimension from 192 to 256 and decrease the number of
attention heads by 1/3. This keeps the training computation and the number of parameters constant
while decreasing the decoding computation. The variant, denoted as MLA-256
    - extend context length from 4K to 200K
    - DSA
    - MTP
  - Ablation Study of Efficient Attention Variants
    - Sliding Window Attention (SWA) Interleave
    - Gated DeltaNet (GDN)
    - We evaluate all methods on four long-context benchmarks: RULER [17], MRCR2, HELMETICL [56], and RepoQA [27]
    - Nevertheless, all of these methods incur an inherent accuracy gap on
fine-grained retrieval tasks—up to 5.69 points on RULER@128K and 7.33 on RepoQA@128K—due
to the unavoidable information loss introduced by efficient attention mechanisms during continualtraining adaptation, even when half of the layers retain full attention. In contrast, DSA is lossless by
construction: its lightning indexer achieves token-level sparsity without discarding any long-range
dependencies, enabling application to all layers with no quality degradation.
  - Pre-training
  - Mid-Training
    - Extended context and training scale.
    - Software engineering data.
    - Long-context data.
    - INT4 Quantization-aware training
  - Post-Training
    - Supervised Fine-Tuning
    - Reasoning RL (GRPO + IcePop)
      - DSA RL insights
      - Mixed domain reasoning RL
    - Agentic RL
    - General RL
    - RL Training Infrastructure: The slime Framework
  - Agentic Engineering
  - Evaluation


