# Visual Backbone Networks
使用 ImageNet-1K classification 和 ImageNet-22K 训练
- Thu, 22 Oct 2020 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
  - ViT
- Mon, 10 Jan 2022 [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  - ConvNext

# CLIP (Contrastive Language-Image Pre-Training)
- Fri, 26 Feb 2021 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
  - CLIP dual-encoder architecture
  - 这么简单的方法效果居然出奇的好，效果让人瞠目结舌
  - openai当时还是那个力大砖飞的openai
- Tue, 9 Nov 2021 [FILIP: Fine-grained Interactive Language-Image Pre-Training](https://arxiv.org/abs/2111.07783):
  - FILIP 和 ColPali 是不是很像 
- Mon, 15 Nov 2021 [LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/abs/2111.07991)
  - Locked-image text Tuning
- Mon, 27 Mar 2023 [EVA-CLIP: Improved Training Techniques for CLIP at Scale](https://arxiv.org/abs/2303.15389)
  - EVA-CLIP
- Mon, 27 Mar 2023 [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
  - Unlike standard contrastive learning with softmax normalization, 
  - the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. 
  - The sigmoid loss simultaneously allows further scaling up the batch size, while also performing better at smaller batch sizes. 
  - 对比学习，batchsize 越大效果越好，32K 差不多饱和
  - The input images are resized to 224×224 resolution
- Wed, 12 Jul 2023 [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
  - To support variable aspect ratios and readily extrapolate to unseen resolutions, we introduce factorized
positional embeddings, where we decompose into separate embeddings ϕx and ϕy of x and y coordinates.
These are then summed together
- Thu, 21 Dec 2023 [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/abs/2312.14238)
  - a vision encoder InternViT-6B
  - a language middleware QLLaMA
    - QLLaMA is developed based on the pre-trained multilingual LLaMA,
and newly added 96 learnable queries and cross-attention
layers (1 billion parameters) that are randomly initialized.
This manner allows QLLaMA to smoothly integrate visual
elements into the language model, thereby enhancing the
coherence and effectiveness of the combined features.
  - Training 
    - stage 1: contrastive pre-training
      - InternViT-6B vs LLaMA-7B -> contrastive loss
      - 等一下，为什么要跟 LLaMA-7B 对比学习，LLaMA-7B 输出的 embed 是用来做采样下一个词的，跟检索没啥关系啊？？？？

# Multimodal Bridge(abstractors)
- Mon, 30 Jan 2023 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)
  - We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM
- Mon, 17 Apr 2023 [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
  - We consider a simple linear layer to connect image features into the word embedding space.
- Mon, 11 Dec 2023 [Honeybee: Locality-enhanced Projector for Multimodal LLM](https://arxiv.org/abs/2312.06742)
  - 高效将M个视觉模态中间状态映射为N个LLM状态
  - C-Abstractor & D-Abstractor

# Vision-Language Models
- Mon, 30 Jan 2023 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)
  - Architecture
    - txt: OPT/FlanT5
    - img: ViT-L/14 from CLIP (Radford et al., 2021) and ViT-g/14 from EVA-CLIP (Fang et al., 2022).
    - Bridge: 
      - We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM
- Mon, 17 Apr 2023 [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
  - Architecture
    - txt: Vicuna
    - img: CLIP visual encoder ViT-L/14
    - Bridge:
      - We consider a simple linear layer to connect image features into the word embedding space.
  - Training
    - Stage 1: Pre-training for Feature Alignment.
      - we keep both the visual encoder and LLM weights frozen, 
      - and maximize the likelihood of (3) with trainable parameters θ = W (the projection matrix) only
    - Stage 2: Fine-tuning End-to-End.
- Thu, 5 Oct 2023 [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)
  - LLaVA-1.5
    - +VQA-v2, +Format prompt, +MLP VL connector, +OKVQA/OCR 
    - +Region-level VQA, +Scale up resolution(336), +GQA,  +ShareGPT, +Scale up LLM 
    - Architecture
      - txt: Vicuna
      - img: CLIP visual encoder ViT-L/14
      - Bridge:
        - we find that improving the vision-language connector’s representation power with a two-layer MLP 
        - can improve LLaVA’s multimodal capabilities, compared with the original linear projection.
  - LLaVA-1.5-HD
- Thu, 24 Aug 2023 [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)
  - Architecture
    - txt: Qwen-7B
    - img: Openclip’s ViT-bigG
    - Bridge:
  - IO
    - 448*448 resolution image
    - The coordinate box is expressed as <box>(x1,y1),(x2,y2)</box>·, 
    - where (x1, y1) and (x2, y2) are normalized values in the range [0, 1000). 
    - Its corresponding text description can be identified by <ref>text_caption</ref>.
- Sat, 3 Aug 2024 [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800)
  - Architecture
    - txt: 
      - MiniCPM 2B & Llama3-Instruct 8B
    - img:  SigLIP SoViT-400m/14 
    - Bridge: 
      - we take advantage of the adaptive visual encoding method proposed by LLaVA-UHD
      - Image Partition & Slice Encoding
      - Token Compression
        - the visual tokens of each slice are compressed into 64 queries for MiniCPM
      - 64 queries for MiniCPM V1&2 and 96 tokens for MiniCPM-Llama3-V 2.5 through this layer.
      - Spatial Schema
  - Training
    - Pre-training
      - Stage-1 224×224，只训练 compression layer
      - Stage-2 224×224 to 448×448，  The whole visual encoder is trained, leaving other parameters frozen
      - Stage-3 The LLM is kept frozen to avoid disruption from the relatively low-quality pre-training data
- Wed, 18 Sep 2024 [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
  - Architecture
    - txt: Qwen2
    - img:  
      - 675M Vision Encoder(DFN’s ViT) + RoPE-2D
      - Naive Dynamic Resolution
      - Multimodal Rotary Position Embedding (M-RoPE)
      - Unified Image and Video Understanding
- Wed, 25 Sep 2024 [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146)
  - Architecture
    - txt: OLMoE-1B-7B, OLMo-7B-1024-preview, Qwen2 7B, Qwen2 72B
    - img: OpenAI’s ViT-L/14 336px CLIP model
    - Evaluation**
      - Broadly speaking, the academic benchmark results and human evaluation agree, with the exception of Qwen2- VL , 
      - which performs strongly on the academic benchmarks and comparatively underperforms in the human evaluation

