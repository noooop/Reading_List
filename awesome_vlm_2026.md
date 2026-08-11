
# The Evolution of VLM Architectures
-  [Vision-Language-Models-Overview](https://github.com/zli12321/Vision-Language-Models-Overview)

VLM design has gone through **four distinct architectural eras** in just six years — and Era 3 has split into two parallel branches. Early models kept frozen vision and language towers, aligned contrastively (CLIP) or bridged by a learnable connector into a frozen LM (BLIP-2, Flamingo). The 2023–2025 generation made a pretrained **LLM the trunk** and treated vision as a bolt-on adapter (LLaVA, Qwen2.5-VL, GPT-4V). The 2025–2026 generation drops the bridge entirely and early-fuses all modalities into **a single transformer** — forking along the *output* axis — and in 2026 the trunk is becoming a **world model** that predicts and acts:

- **Era 3a — Native Multimodal Input → Text Out.** Image, video, and (sometimes) audio enter a single early-fused token stream, but generation is still autoregressive text. This is the design used by today's general-purpose flagships: **Qwen3.5 / Qwen3.6, Gemma 4, Gemini 3, GPT-5.4, Phi-4-Reasoning-Vision, Claude Opus 4.6, Nemotron 3 Nano Omni**.
- **Era 3b — Omni-Modal Unified I/O.** The same fused trunk plus dedicated **image / video decoder** (VAE / DiT / flow-matching) and/or **audio codec** decoder heads, so the model can also *generate* images, video, and speech — via autoregression or, increasingly, **discrete diffusion / AR-Diffusion** (LLaDA2.0-Uni, Mamoda2.5). This is the design used by unified models: **BAGEL, Qwen3.5-Omni, InternVL-U, Emu3 / Emu3.5, Erin 5.0, DeepSeek-Janus-Pro, LLaDA2.0-Uni, Mamoda2.5**. Generation-only specialists share the same decoder stack without the understanding half — **Sora 2, Veo 3, Kling** now generate video with **synchronized audio**, and they double as the substrate for Era 4 world models (DreamX-World builds on Wan, OmniDreams on Cosmos).
- **Era 4 — World-Action Models (2026 →).** The unified trunk adds **action** as a first-class modality and closes the loop with the environment: it predicts future observations, maintains persistent state and spatial memory, and emits actions — **generator, perceiver, and policy in one network**: **Cosmos 3, Kairos, DreamX-World 1.0, OmniDreams** (see [§1.1 World Models](#worldmodels)).

<img src="https://github.com/zli12321/Vision-Language-Models-Overview/raw/main/assets/vlm_architecture_evolution.svg" width="400">

我的VLM知识停留在 Era 2 ~ Era 3 需要补充一下新知识了！！

## Era 1: Contrastive / bridged Two(vision and language) towers

## Era 2: LLM backbone

### Qwen
- Thu, 28 Sep 2023 [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
  - https://huggingface.co/collections/Qwen/qwen
- Thu, 24 Aug 2023 [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)
  - Architecture
    - txt: Qwen-7B                                                                        7.7B
    - img:                                                                                1.9B
      - Openclip’s ViT-bigG, input images are resized to a specific resolution         
    - projector:                                                                          0.08B
      - Learnable Query Embs + CrossAttn + 2D absolute positional encodings
      - This mechanism compresses the visual feature sequence to a fixed length of 256
- Sun, 4 Feb 2024 [Introducing Qwen1.5](https://qwen.ai/blog?id=qwen1.5)
  - https://huggingface.co/collections/Qwen/qwen15
- Mon, 15 Jul 2024 [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
  - https://huggingface.co/collections/Qwen/qwen2
- Mon, 15 Jul 2024 [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
  - https://huggingface.co/collections/Qwen/qwen2-audio
- Wed, 18 Sep 2024 [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
  - https://huggingface.co/collections/Qwen/qwen2-vl
- Thu, 19 Dec 2024 [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
  - https://huggingface.co/collections/Qwen/qwen25
- Sun, 26 Jan 2025 [Qwen2.5-1M Technical Report](https://arxiv.org/abs/2501.15383)
  - https://huggingface.co/collections/Qwen/qwen25-1m
- Wed, 19 Feb 2025 [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
  - https://huggingface.co/collections/Qwen/qwen25-vl
- Wed, 26 Mar 2025 [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)
  - https://huggingface.co/collections/Qwen/qwen25-omni
- Wed, 14 May 2025 [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
  - https://huggingface.co/collections/Qwen/qwen3
- Wed, 9 Sep 2025 [Qwen3-Next](https://huggingface.co/collections/Qwen/qwen3-next)
- Mon, 22 Sep 2025 [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)
  - https://huggingface.co/collections/Qwen/qwen3-omni
- Wed, 26 Nov 2025 [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
  - https://huggingface.co/collections/Qwen/qwen3-vl
- Thu, 29 Jan 2026 [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
  - https://huggingface.co/collections/Qwen/qwen3-asr
- Sun, 15 Feb 2026 [Qwen3.5: Towards Native Multimodal Agents](https://qwen.ai/blog?id=qwen3.5)
  - https://huggingface.co/collections/Qwen/qwen35
- Tue, 14 Apr 2026 [Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
  - https://huggingface.co/collections/Qwen/qwen36

## Era 3a — Native Multimodal Input

## Era 3b — Omni-Modal Unified I/O

## Era 4 — World-Action Models