
# VLMs
- Wed, 18 Sep 2024 [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)
- Fri, 27 Sep 2024 [MinerU: An Open-Source Solution for Precise Document Content Extraction](https://arxiv.org/abs/2409.18839)
  - opendatalab/MinerU2.0-2505-0.9B
  - Built on Qwen2 + siglip-so400m-patch14-384
- Wed, 19 Feb 2025 [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- Tue, 25 Feb 2025 [olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models](https://arxiv.org/abs/2502.18443)
  - Built on Qwen2-VL-7B-Instruct
  - We use single node with 8 x NVIDIA H100 (80GB) GPUs. A single training run took 16 node hours, with all training experiments totaling 365 node hours
- Tue, 20 May 2025 [Dolphin: Document Image Parsing via Heterogeneous Anchor Prompting](https://arxiv.org/abs/2505.14059)
  - Swin Transformer + mBart
- Sun, 1 Jun 2025 [Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing](https://arxiv.org/abs/2506.03197)
  - Built on Qwen/Qwen2.5-VL-8B-Instruct
  - optimized via Group Relative Policy Optimization (GRPO)
    - Edit Distance Reward (Rdist)
    - Count Reward (Rcount)
    - Order Reward (Rorder)
  - We fine-tune the Qwen2.5-VL-7B model using GRPO within a distributed training setup based on Verl [45, 66], utilizing 8 A100 GPUs (80GB).
- Thu, 5 Jun 2025 [MonkeyOCR: Document Parsing with a Structure-Recognition-Relation Triplet Paradigm](https://arxiv.org/abs/2506.05218)
  - Structure Detection, Content Recognition, Relation Prediction
  - Built on Qwen/Qwen2.5-VL-3B-Instruct
  - Our 3B model was trained for 53 hours on 32 A800 GPUs.
- Thu, 12 Jun 2025 [OCRFlux-3B](https://huggingface.co/ChatDOC/OCRFlux-3B)
  - Built on Qwen/Qwen2.5-VL-3B-Instruct
- Wed, 30 Jul 2025 [dots.ocr: Multilingual Document Layout Parsing in a Single Vision-Language Model](https://huggingface.co/rednote-hilab/dots.ocr)
  - Built on Qwen/Qwen2.5-VL-3B-Instruct
- Fri, 1 Aug 2025 [DocTron-Formula: Generalized Formula Recognition in Complex and Structured Scenarios](https://arxiv.org/abs/2508.00311)
  - Built on Qwen/Qwen2.5-VL-8B-Instruct
- Mon, 18 Aug 2025 [DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool Interleaved Vision-Language Model](https://arxiv.org/abs/2508.13238)
  - Built on Qwen/Qwen2.5-VL-7B-Instruct
  - on a single node with 8 NVIDIA A100 GPUs
  - four-stage process
    - think: it starts with its own OCR read
    - tool : then consults specialized external tools for a second opinion
    - rethink : re-examines the image with that extra context
    - answer : finally delivers its response

# Benchmark
- Tue, 10 Dec 2024 [OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations](https://arxiv.org/abs/2412.07626)