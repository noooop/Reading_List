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
- Wed, 12 Jul 2023 [Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
  - To support variable aspect ratios and readily extrapolate to unseen resolutions, we introduce factorized
positional embeddings, where we decompose into separate embeddings ϕx and ϕy of x and y coordinates.
These are then summed together

# Vision-Language Models
- Mon, 30 Jan 2023 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)
  - We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM
  - BLIP-2 OPT/FlanT5 + ViT 
  - ViT-L/14 from CLIP (Radford et al., 2021) and (2) ViT-g/14 from EVA-CLIP (Fang et al., 2022).
- Thu, 24 Aug 2023 [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)
  - Qwen-7B +  Openclip’s ViT-bigG
    - Qwen-VL & Qwen-VL-Chat & (Qwen-VL-Plus & Qwen-VL-Max) (404)
  - IO
    - 448*448 resolution image
    - The coordinate box is expressed as <box>(x1,y1),(x2,y2)</box>·, 
    - where (x1, y1) and (x2, y2) are normalized values in the range [0, 1000). 
    - Its corresponding text description can be identified by <ref>text_caption</ref>.
- Wed, 18 Sep 2024 [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
  - Qwen2 + 675M Vision Encoder(DFN’s ViT) + RoPE-2D
  - Naive Dynamic Resolution
  - Multimodal Rotary Position Embedding (M-RoPE)
  - Unified Image and Video Understanding
- Wed, 25 Sep 2024 [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](https://arxiv.org/abs/2409.17146)
  - img: OpenAI’s ViT-L/14 336px CLIP model
  - txt: 
    - OLMoE-1B-7B, OLMo-7B-1024-preview, Qwen2 7B, Qwen2 72B
  - Evaluation
    - Broadly speaking, the academic benchmark results and human evaluation agree, with the exception of Qwen2- VL , 
    - which performs strongly on the academic benchmarks and comparatively underperforms in the human evaluation

