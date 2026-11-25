<div align="center">
  <img src="assets/header.png" alt="Andrew Rober — ML & High-Performance Systems Researcher" width="100%" />
  <br /><br />
  <a href="https://andrewrober.github.io/"><img src="https://img.shields.io/badge/Research%20Portal-andrewrober.github.io-38bdf8?style=flat-square" alt="Research Portal" /></a>
  <a href="https://www.linkedin.com/in/andrew-rober/"><img src="https://img.shields.io/badge/LinkedIn-Andrew%20Rober-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://huggingface.co/AndrewR123"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models%20%26%20Spaces-ffb000?style=flat-square" alt="HuggingFace" /></a>
  <a href="mailto:andrew.rober@bytesglue.com"><img src="https://img.shields.io/badge/Email-andrew.rober%40bytesglue.com-ea4335?style=flat-square&logo=gmail&logoColor=white" alt="Email" /></a>
  <a href="https://bytesglue.com"><img src="https://img.shields.io/badge/Company-BytesGlue-10b981?style=flat-square" alt="BytesGlue" /></a>
</div>

---

### Research Overview

I design **mathematically grounded machine learning algorithms** and **bare-metal GPU / SIMD systems**—focusing on structural bottlenecks in **LLM pretraining & tokenization**, **mechanistic interpretability**, **non-asymptotic high-dimensional statistics**, and **stochastic optimization**. Every research release pairs closed-form theoretical guarantees with reproducible from-scratch PyTorch/CUDA and vectorized `.NET` implementations.

> **Open to Opportunities & Collaboration:** Available for **Senior Research Engineer / Member of Technical Staff (LLM Pretraining, Post-Training RLVR, Inference & Systems)** roles and high-impact AI/systems consulting (Dubai, UAE · Remote · Relocation) — reach me directly at **[andrew.rober@bytesglue.com](mailto:andrew.rober@bytesglue.com)**.

---

### Selected Research Papers & Open-Source Implementations

| Research Monograph & Codebase | Core Mathematical & Empirical Contributions | Artifacts |
| :--- | :--- | :---: |
| **[ResidualRescaledBF16](https://github.com/AndrewRober/ResidualRescaledBF16)**<br/>*Breaking the 7-Bit BF16 Mantissa Ceiling (`CSP-BF16-IR`)* | **Intralayer-free mixed-precision spectral Volterra–Newton solver** recovering `24-bit IEEE-FP32` trajectory accuracy on unmodified `7-bit BF16` Tensor Cores (`NVIDIA GB10` & `RTX 5060`).<br/>• **Antipodal Residual Rescaling** ($`\mathcal{O}(\varepsilon_{\mathrm{bf16}}^{2/3})`$, `3.68x` single-sweep JVP gain at `0` extra FLOPs)<br/>• **`46.9M` SwiGLU DiT (`PTX` + `CUDAGraph`):** `9.37e-8` `L2` error (`>23,000x` beyond `ParaDiGMS`'s `BF16` wall; **`44.2x` faster** than `CG-RK4`; `170x` in `ParaFlow-BF16`)<br/>• **Root-Time `z = sqrt(tau)` Diffusion Desingularization:** Eliminates `152.2x` `1/alpha_T` `BF16` blowup; lifts official `google/ddpm-cifar10-32` (`69.69 dB`, `86.08 dB` vs. Colloc) & `64x`-larger `256x256` `google/ddpm-celebahq-256` (`113.7M`, `+24.5 dB` gain, beating `FP32` `DPM-Solver-2-100`) | [![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](https://github.com/AndrewRober/ResidualRescaledBF16/blob/main/paper/main.pdf)<br/>[![HF](https://img.shields.io/badge/%F0%9F%A4%97-Hub-ffb000?style=flat-square)](https://huggingface.co/AndrewR123/ResidualRescaledBF16)<br/>[![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=flat-square)](https://github.com/AndrewRober/ResidualRescaledBF16/releases/tag/v1.0.0) |
| **[MechanisticPhraseTokenization](https://github.com/AndrewRober/MechanisticPhraseTokenization)**<br/>*What Earns a Phrase Its Own Token?* | **34 from-scratch pretrained Transformers (`30.4M → 149.6M` zero-shot transfer)** isolating superword vocabulary allocation.<br/>• **33%–37% Slot-Waste Audit** of iterative `SuperBPE` & `SupraTok`<br/>• **Prefix-Cone Decoding** (`+0.80%` to `+1.59%` next-word recovery with `0` retraining)<br/>• **Foldable Compositional-Residual Superwords** ($`W_P = \bar{W}_t + \Delta_P`$, `0.0` float folding error, `#1` OOD `BPB`, unsupervised idiom detection)<br/>• **Attention-Query De-Multiplexing** ($`D_{\text{bind}}`$) **& Compute Gearshifting** (`-51.2%` Layer-2 stitching; `86.81%` `BLiMP` & `+1.58%` CoT-free test-time scaling at `150M`) | [![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square)](https://github.com/AndrewRober/MechanisticPhraseTokenization/blob/main/paper/main.pdf)<br/>[![HF](https://img.shields.io/badge/%F0%9F%A4%97-Weights-ffb000?style=flat-square)](https://huggingface.co/AndrewR123/MechanisticPhraseTokenization)<br/>[![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=flat-square)](https://github.com/AndrewRober/MechanisticPhraseTokenization/releases/tag/v1.0.0) |
| **[KurtosisScaleCorrection](https://github.com/AndrewRober/KurtosisScaleCorrection)**<br/>*Beyond Bessel's Correction & Empirical-Bayes GRPO* | Power-scale concavity–convexity Jensen duality, **Kurtosis-Optimal Scale Shrinkage (KPS)** for heavy-tailed distributions, and **Empirical-Bayes Advantage Normalization for GRPO** LLM reinforcement learning (`PyTorch CUDA` + `C#/.NET 8` + `NVIDIA GB10` LLM benchmarks). | [![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=flat-square)](https://github.com/AndrewRober/KurtosisScaleCorrection/releases/tag/v1.0.0) |
| **[IntrinsicDimKMeans](https://github.com/AndrewRober/IntrinsicDimKMeans)**<br/>*Spectral Intrinsic-Dimension MDL K-Means* | **Zero-hyperparameter cluster count** ($`K^*`$) **discovery** via Random Matrix Theory (Marchenko–Pastur bulk edge) and Minimum Description Length on the active spectral manifold—overcoming high-D Transformer embedding collapse and Voronoi overfitting (`Python` & `C#/.NET`). | [![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=flat-square)](https://github.com/AndrewRober/IntrinsicDimKMeans/releases/tag/v1.0.0) |
| **[HigherDimensionalEWMABiasCorrection](https://github.com/AndrewRober/HigherDimensionalEWMABiasCorrection)**<br/>*Non-Stationary High-D EWMA & AdaTCAdamW* | Exact non-stationary drift bias elimination in higher-dimensional exponential moving averages, $`O(1/d)`$ blessing-of-dimensionality variance bounds, `PyTorch` **AdaTCAdamW** optimizer, and hardware-accelerated `C#/.NET` SIMD vector engine. | [![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=flat-square)](https://github.com/AndrewRober/HigherDimensionalEWMABiasCorrection/releases/tag/v1.0.0) |

---

### Featured Results: Breaking the 7-Bit `BF16` Mantissa Ceiling (`256x256` Diffusion) & Mechanistic Superword Tokenization

<div align="center">
  <a href="https://github.com/AndrewRober/ResidualRescaledBF16">
    <img src="https://raw.githubusercontent.com/AndrewRober/ResidualRescaledBF16/main/paper/figures/fig6_celebahq256_scaleup.png" alt="Zero-Shot 64x Scale-Up on google/ddpm-celebahq-256 (113.67M UNet)" width="96%" />
  </a>
  <br /><br />
  <a href="https://github.com/AndrewRober/MechanisticPhraseTokenization">
    <img src="https://raw.githubusercontent.com/AndrewRober/MechanisticPhraseTokenization/main/paper/figures/fig6_scaleup_300m_150m.png" alt="300M 3-Seed Confirmation & 150M Compute Gearshifting" width="96%" />
  </a>
</div>

---

### Open-Source AI Safety & Systems Engineering Libraries

| Repository | Description | Stack |
| :--- | :--- | :---: |
| **[CodexCli-GuardRails](https://github.com/AndrewRober/CodexCli-GuardRails)** | Deterministic action-risk classification, workspace containment, secret redaction, and tamper-evident approval gates for AI coding agents via **MCP Server** (`mcp-server/`) and multi-agent `SKILL.md` policies. | `Python` · `MCP` · `AI Safety` |
| **[SIMD-Tutorial-Benchmark](https://github.com/AndrewRober/SIMD-Tutorial-Benchmark)** | Hardware-accelerated C# / `.NET` SIMD vectorization benchmarks and throughput profiling. | `C#` · `.NET SIMD` |
| **[SortIO](https://github.com/AndrewRober/SortIO)** | Lightweight C# sorting library implementing 30+ classical and hybrid sorting algorithms with test suites. | `C#` · `Algorithms` |
| **[Aurora-Samples](https://github.com/AndrewRober/Aurora-Samples)** | Reference integration projects and healthcare EDI/HIPAA workflows for HIPAAsuite Aurora libraries. | `C#` · `.NET` |

---

### Technical Stack & Hardware Environments

- **Deep Learning & LLM Pretraining:** `PyTorch 2.x`, `torch.compile` (Inductor / Triton), `FlashAttention`, Fused Chunked Cross-Entropy, Custom Activation/Gradient Hooks, `HuggingFace Tokenizers` & `Datasets`
- **High-Performance Systems:** `NVIDIA Blackwell (GB10)` Unified-Memory Profiling, `CUDA` Mixed Precision (`BF16` / `FP8`), `C# / .NET 8` `System.Numerics` / `Vector512` SIMD Engines
- **Theory & Quantitative Modeling:** Information Theory, Random Matrix Theory (`RMT`), Minimum Description Length (`MDL`), Empirical Bayes & Shrinkage Estimation, Non-Stationary Time-Series Optimization
