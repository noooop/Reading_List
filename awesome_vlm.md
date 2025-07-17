# Visual Backbone Networks
使用 ImageNet-1K classification 和 ImageNet-22K 训练
- Thu, 22 Oct 2020 [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
  - ViT
- Mon, 10 Jan 2022 [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
  - ConvNext

# CLIP (Contrastive Language-Image Pre-Training)
- Fri, 26 Feb 2021 [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
  - CLIP
- Mon, 27 Mar 2023 [EVA-CLIP: Improved Training Techniques for CLIP at Scale](https://arxiv.org/abs/2303.15389)
  - EVA-CLIP

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
  