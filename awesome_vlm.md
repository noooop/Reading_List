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

# Multimodal Projector(abstractors)
- Mon, 30 Jan 2023 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)
  - We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM
- Mon, 17 Apr 2023 [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
  - We consider a simple linear layer to connect image features into the word embedding space.
- Mon, 11 Dec 2023 [Honeybee: Locality-enhanced Projector for Multimodal LLM](https://arxiv.org/abs/2312.06742)
  - 高效将M个视觉模态中间状态映射为N个LLM状态
  - C-Abstractor & D-Abstractor

# High-resolution LMMs
- Mon, 18 Mar 2024 [LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images](https://arxiv.org/abs/2403.11703)
  - High-Resolution Image Partition Strategy
  - Arbitrary Aspect Ratio Slice Encoding
  - Compression Layer
  - Spatial Schema for Image Slices
- Wed, 18 Dec 2024 [LLaVA-UHD v2: an MLLM Integrating High-Resolution Semantic Pyramid via Hierarchical Window Transformer](https://arxiv.org/abs/2412.13871)
  - Hierarchical window (Hiwin) transformer

# Vision-Language Models
- Mon, 30 Jan 2023 [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/pdf/2301.12597)
  - Architecture
    - txt: OPT/FlanT5
    - img: ViT-L/14 from CLIP (Radford et al., 2021) and ViT-g/14 from EVA-CLIP (Fang et al., 2022).
    - projector: 
      - We propose Q-Former as the trainable module to bridge the gap between a frozen image encoder and a frozen LLM
- Mon, 17 Apr 2023 [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
  - Architecture
    - txt: Vicuna
    - img: CLIP visual encoder ViT-L/14
    - projector:
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
      - projector:
        - we find that improving the vision-language connector’s representation power with a two-layer MLP 
        - can improve LLaVA’s multimodal capabilities, compared with the original linear projection.
  - LLaVA-1.5-HD(AnyRes)
    - Dynamic High Resolution (split & resize)
    - we overcome this by dividing the image into smaller image patches of the resolution that the vision encoder is originally trained for, and encode them independently. 
    - After obtaining the feature maps of individual patches, we then combine them into a single large feature map of the target resolution, and feed that into the LLM.
    - To provide the LLM with the global context and to reduce the artifact of the split-encode-merge operation, 
    - we additionally concatenate the feature of a downsampled image to the merged feature map.
    - This allows us to scale the input to any arbitrary resolution and maintain the data efficiency of LLaVA-1.5. 
    - We call this resulting model LLaVA-1.5-HD.
- Thu, 24 Aug 2023 [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)
  - Architecture
    - txt: Qwen-7B
    - img: Openclip’s ViT-bigG
    - projector:
  - IO
    - 448*448 resolution image
    - The coordinate box is expressed as <box>(x1,y1),(x2,y2)</box>·, 
    - where (x1, y1) and (x2, y2) are normalized values in the range [0, 1000). 
    - Its corresponding text description can be identified by <ref>text_caption</ref>.
- Tue, 9 Apr 2024 [InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD](https://arxiv.org/abs/2404.06512)
  - Architecture
    - txt: InternLM2-7B
    - img: CLIP visual encoder ViT-L/14(336×336)
    - projector:
      - Dynamic Image Partition + Global-Local Format + Image 2D Structure Newline Indicator.
  - Dive into Resolution
    - High-Resolution Training is Critical for HD-OCR tasks.
    - Higher Inference Resolution Leads to better results on Text-related Tasks.
  - High-Resolution Strategy Ablation
    - The Role of Global-View
    - The Role of the Newline Token
    - Influence of Token Merging Strategy
- Mon, 22 Apr 2024 [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/abs/2404.14219)
  - Phi-3.5-Vision 
  - Architecture
    - txt: phi-3.5-mini
    - img: CLIP ViT-L/14
    - projector:
      - dynamic cropping strategy [DZZ+24b] is utilized to split the input image into a 2d array of blocks
      - InternLM-XComposer2-4KHD
- Tue, 30 Apr 2024 [LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
  - 使用相同的303.5M Vision Encoder，更新更大的 Qwen1.5-110B， Qwen1.5-72B，LLaMA3-8B， 效果就是好
  - 多模态MMMU 和 单模态 MMLU & 模型大小非常相关
- Mon, 18 Mar 2024 [LLaVA-UHD: an LMM Perceiving Any Aspect Ratio and High-Resolution Images](https://arxiv.org/abs/2403.11703)
    - Architecture (LLaVA-1.5)
      - txt: Vicuna
      - img: CLIP visual encoder ViT-L/14
      - projector:
        - we compress the visual tokens of each image slice using a shared perceiver resampler layer,
    - Modularized Visual Encoding
      - High-Resolution Image Partition Strategy
      - Arbitrary Aspect Ratio Slice Encoding
      - Compression Layer
      - Spatial Schema for Image Slices
    - Ablation Study
      -  (1) We replace
the padding strategy of LLaVA-1.5 with the adaptive encoding strategy of LLaVA-UHD, supporting
arbitrary aspect ratios while maintaining identical maximum resolutions. We can observe consistent
improvement since wasted computation from padding is avoided.
      -  (2) We replace the perceiver
resampler of LLaVA-UHD with the 2-layer MLP of LLaVA-1.5. We observe that perceiver resampler
achieves comparable or better performance than MLP, using only 12.9% computation cost.
      -  (3) We
further replace the LLaVA-UHD image partition strategy with the naive partition strategy [24] (i.e.,
fixed 2 × 2 slices). Results show that LLaVA-UHD can more properly divide images into slices for better performance.
      -  (4) We remove the spatial schema from LLaVA-UHD. The performance
degradation demonstrates the effectiveness and necessity of spatial schema in informing the dynamic
slice positions for LMMs.
- Mon, 24 Jun 2024 [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](https://arxiv.org/abs/2406.16860)
  - 内容太丰富了，但最重要的一点是使用多个 vision feature extractor 互补
  - 比如 CLIP 视觉和文本对齐，OpenCLIP ConvNeXt-XXL@1024 支持高分辨率，DINOv2 ViT-L/14@518 视觉任务比较强
- Sat, 3 Aug 2024 [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](https://arxiv.org/abs/2408.01800)
  - Architecture
    - txt: 
      - MiniCPM 2B & Llama3-Instruct 8B
    - img: SigLIP SoViT-400m/14 
    - projector: 
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
- Wed, 28 Aug 2024 [Eagle: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders](https://arxiv.org/abs/2408.15998)
  - STRONGER CLIP ENCODER
    - Direct interpolation (CLIP encoder) to 448 × 448 can achieve competitive performance while being more efficient
  - VISION EXPERTS
    - experts
      - (1) Vision-Language Alignment: CLIP/ConvNeXt/OpenCLIP
      - (2) Object-Centric: EVA-02
      - (3) OCR: Pix2Struct
      - (4) Segmentation: SAM
      - (5) Self-supervised: DINOv2
    - distinct advantages of different experts
      - We resize the output 2D feature maps of each vision encoder using bilinear interpolation or 
      - pixel shuffle (Shi et al., 2016) to ensure that the visual token number equals 1024.
      - unfreezing the vision experts again leads to consistent improvement,
        - MLLMs with these task specific vision encoders achieve optimal performance in their pretraining domains
  - FUSION STRATEGY
    - (1) Sequence Append
    - (2) Channel Concatenation
    - (3) LLaVA-HR: injecting highresolution features into low-resolution vision encoders using mixture-of-resolution adapter
    - (4) Mini-Gemini: using the CLIP tokens as the low-resolution queries to cross-attend another high-resolution vision encoder in the co-located local windows
    - (5) Deformable Attention
    - Channel Concatenation stands out with the best performance, expandability, and efficiency.
  - VISON-LANGUAGE PRE-ALIGNMENT
    - 1) training each pre-trained vision expert with their own projector, while keeping the language model frozen; 
    - 2) combining all vision experts from the first step and training both the projector and vision experts; 
    - 3) training the whole model on SFT data.
  - EXTENSION TO MULTI-EXPERTS
    - introducing additional vision encoders enhances the performance
    - CLIP-448 + ConvNext-1024 组合作为baseline
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
- Wed, 18 Dec 2024 [LLaVA-UHD v2: an MLLM Integrating High-Resolution Semantic Pyramid via Hierarchical Window Transformer](https://arxiv.org/abs/2412.13871)
  - projector: 
    - we present LLaVA-UHD v2, an MLLM with advanced
perception abilities by introducing a well-designed vision-language projector, the
Hierarchical window (Hiwin) transformer. Hiwin transformer enhances MLLM’s
ability to capture diverse multi-modal visual granularities, by incorporating our
constructed high-resolution semantic pyramid. 
- Wed, 19 Feb 2025 [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- Mon, 20 Jan 2025 [Eagle 2: Building Post-Training Data Strategies from Scratch for Frontier Vision-Language Models](https://arxiv.org/abs/2501.14818)
  - Architecture
    - txt: Qwen2.5-7B
    - img: SigLIP 448*448 + ConvNeXt 512*512

# Knowledge Distillation
- Sun, 10 Dec 2023 [AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One](https://arxiv.org/abs/2312.06709)
  - CLIP, DINOv2, and SAM
- Tue, 10 Dec 2024 [RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models](https://arxiv.org/abs/2412.07679)
  - Challenges
    - 3.1. Achieving Multi-Resolution Robustness
      - where feature distributions shift significantly based on input resolution 
      - Specifically, low-resolution inputs yield DINO-like features, 
      - while high-resolution inputs produce SAM-like features
      - We trace this behavior to the student learning from different teachers at different resolutions during training
    - 3.2. Token Count
      - an excessive number of vision tokens can negatively impact performance or lead to sequence overflows
  - Method
    - Finding 1. High-resolution inference through tiling causes the vision encoder to lose global context and exhibit poor scaling equivariance.
    - Finding 2. For the student model to be consistently accurate across resolutions, it is sufficient to match all teachers at all resolutions, and to train at two resolutions simultaneously in the final training stage.
    - Finding 3. Mosaic augmentation greatly reduces the training cost associated with learning from high-resolution teachers and eliminates the need for feature interpolation. Student quality is even improved with this optimization.
    - Finding 4. PHI Standardization helps balance the energy spent learning from each teacher.
    - Finding 5. All teachers are beneficial, including SAM, despite recent trends. It also has broad downstream applicability, granting our student the same abilities.
    - Finding 6. Minimizing the number of partitions seems to be beneficial, assuming you can afford the teacher overhead. Under compute constraints, partitioning is an effective strategy to reduce the overhead.
    - SigLIP Teacher
      - Our choice is validated by the significant improvements observed in VLM tasks
    - Finding 7. Token Merging is very effective at retaining the most diverse information under high compression ratios.
    - Finding 8. Intermediate layer activations greatly benefit downstream tasks if a non-linear transformation is employed.
  - Ablation Studies
    - 𝒜: RADIOv2.1-L*: Baseline
    - ℬ: 𝒜 + multi-res: Eliminate modes
    - 𝒞: ℬ - OpenAICLIP + SigLIP: Better VLM
    - 𝒟: 𝒞 + ViT-H: Bigger backbone
    - ℰ: 𝒟 + Token Merging: Improve VLM

# Token Merging
- Thu, 30 Mar 2023[Token Merging for Fast Stable Diffusion](https://arxiv.org/abs/2303.17604)