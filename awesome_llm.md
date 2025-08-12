
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