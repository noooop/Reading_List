
# KV Cache Transfer

## P/D Disaggregated
- Mon, 24 Jun 2024 [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)

## MLA
- Tue, 7 May 2024 [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
  - MLA (reducing KV cache by 93.3%)
  - MOE
- Fri, 27 Dec 2024 [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  - Inference and Deployment
    - To simultaneously ensure both the Service-Level Objective (SLO) for online services and high throughput, we employ the following deployment strategy that separates the prefilling and decoding stages.

## Hybrid Attention(KDA/SWA/GDN/Lightning...)
随着 linear-complexity block 加入，kv cache 大小成数量级相机，跨机房传输kv cache 成为可能
- Thu, 16 Apr 2026 [Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter](https://arxiv.org/abs/2604.15039)
  - The Bandwidth Wall in Conventional PD Disaggregation
  - Hybrid Attention Changes the PD Deployment Boundary
    - Hybrid Prefix Cache Pool
  - From Intra-Datacenter PD to Prefill-as-a-Service Paradigm
    - Intra-Datacenter 400Gbps~800Gbps vs Cross-Datacenter 100Gbps
  - Dual-Timescale Scheduling
    足够长的prefill才需要offloading到远端处理。
    - Short-term: bandwidth- and cache-aware routing. 
    - Long-term: traffic-driven allocation re-optimization. 
  - Discussion
    - KVCache-friendly model architecture.
    - KVCache compression and reuse. 
    - Phase-specialized inference hardware. 

