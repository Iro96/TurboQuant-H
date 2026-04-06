
# TurboQuant-H: Hierarchical Adaptive KV Cache Compression Beyond TurboQuant

## Abstract

TurboQuant introduces a data-oblivious, two-stage vector quantization scheme for large language model key–value (KV) cache compression, combining random rotation plus scalar quantization with a 1-bit Quantized Johnson–Lindenstrauss (QJL) residual to preserve both mean-squared error (MSE) and inner-product fidelity. This paper proposes **TurboQuant-H**, a stronger hybrid design that keeps the mathematical core of TurboQuant while addressing its practical limitations: rotation overhead, fixed bit allocation, lack of transformer structure awareness, and limited compression headroom. TurboQuant-H combines (i) structured orthogonal preprocessing, (ii) head-wise adaptive bit allocation, (iii) saliency-aware token retention, (iv) low-rank residual correction, and (v) optional temporal eviction for stale KV entries. The central claim is not that quantization alone can reliably deliver 20–30× compression, but that a **hybrid cache policy** can target that regime while preserving quality better than aggressive uniform low-bit quantization. We provide a complete formulation, a PyTorch benchmark harness for SmolLM-135M-Instruct, and an evaluation protocol for memory, perplexity, and generation quality.

## 1. Background

Google Research’s TurboQuant is explicitly motivated by the KV-cache bottleneck in transformer inference and vector search. The source method uses random rotation to induce a concentrated coordinate distribution and then applies scalar quantization, followed by a 1-bit QJL residual stage to remove bias in inner-product estimation. Google reports quality-neutral KV cache quantization at 3.5 bits per channel, marginal degradation at 2.5 bits per channel, and at least 6× memory reduction in needle-in-a-haystack benchmarks. The arXiv abstract states that TurboQuant is data-oblivious, online, and near-optimal up to a small constant factor relative to the distortion lower bound. Hugging Face’s SmolLM-135M-Instruct model card also warns that 4-bit quantization degrades quality for the 135M and 360M variants, so a strong benchmark must test beyond naive uniform quantization.

## 2. Why TurboQuant Needs a New Design

TurboQuant is strong when the objective is purely vector fidelity, but KV cache quality depends on more than MSE:

- attention logits stability,
- token recency,
- head-specific anisotropy,
- outlier channels,
- and the fact that not all cached tokens are equally useful.

A pure bit-width reduction usually plateaus around 4–8× for practical quality-preserving settings. To reach the 20–30× regime, the cache policy must exploit **structure**: keep a small high-precision window, compress older tokens much more aggressively, and provide a cheap correction term for the remaining approximation error.

## 3. TurboQuant-H

Let the KV cache for a layer be tensors \(K, V \in \mathbb{R}^{H \times T \times D}\), where \(H\) is the number of heads, \(T\) the sequence length, and \(D\) the head dimension.

TurboQuant-H compresses each layer with five components:

1. **Structured orthogonal preprocessing**  
   Use a fast block transform \(R\) (Hadamard + random sign, or a learned diagonal whitening approximation) to flatten heavy tails:
   \[
   \tilde{K} = RK,\quad \tilde{V} = RV.
   \]

2. **Head-wise adaptive quantization**  
   Each head gets a bit budget \(b_{l,h}\) chosen from a small set, rather than a single global width. Older or less salient heads receive fewer bits.

3. **Saliency-aware token retention**  
   Keep a small critical window in fp16 and identify additional salient tokens using a score such as
   \[
   s_t = \alpha \|k_t\|_2 + \beta \|v_t\|_2 + \gamma \,\mathrm{age}^{-1}(t).
   \]
   Only the most useful tokens stay in full precision.

4. **Low-rank residual correction**  
   For the quantized old segment, store a compact correction:
   \[
   R_{\text{res}} \approx u v^\top
   \]
   or, in the benchmark implementation, a per-head residual bias vector. This is cheap but noticeably improves reconstruction.

5. **Temporal eviction / summarization**  
   Tokens older than a configurable horizon are either quantized at the lowest bit-width or dropped into a summary cache. This is the main mechanism that makes 20–30× feasible.

## 4. Objective

TurboQuant-H minimizes
\[
\mathcal{L} = \lambda_1 \|K-\hat K\|_2^2 + \lambda_2 \|V-\hat V\|_2^2
\]

- \lambda_3 \,\mathrm{KL}\big(\mathrm{softmax}(QK^\top/\sqrt{D}) \,\|\, \mathrm{softmax}(Q\hat K^\top/\sqrt{D})\big)
- \lambda_4 \,\mathrm{Bits}(\hat K,\hat V),

subject to a hard memory budget.

The attention term is the critical extension beyond TurboQuant: it optimizes the object that actually matters in generation, not just the vector norm.

## 5. Why 20–30× Is Possible Only in Hybrid Form

For fp16 KV cache, the baseline cost is roughly \(16\) bits per element. A quantizer alone usually cannot drop the average below \(\sim 2\) bits per element without hurting quality. Therefore, **20–30×** requires a second lever: **reducing how much cache is kept**. TurboQuant-H does this via recency-aware retention and summarization.

If only a fraction \(\rho\) of old tokens are retained in compressed form and the rest are evicted or summarized, the effective memory can approach
\[
\bar b \approx (1-\rho)b_q + \rho b_{\text{summary}},
\]
where \(b_q\) is the low-bit representation and \(b_{\text{summary}}\) is the cost of a summary state. This is the path to 20–30×; pure quantization is not enough.

## 6. Implementation Notes

The accompanying Python benchmark:

- loads `HuggingFaceTB/SmolLM-135M-Instruct`,
- runs a manual generation loop,
- extracts `past_key_values`,
- compresses keys and values layer-by-layer,
- reconstructs them before the next token step,
- and reports:
  - estimated compressed memory,
  - compression ratio,
  - generation output.

The code is CPU-safe and GPU-safe. It uses a deterministic uniform quantizer plus optional Hadamard preprocessing and a residual correction term. This is intentionally simple enough to test quickly while still reflecting the TurboQuant-H design.

## 7. Experimental Protocol

Recommended evaluation:

- **Prompt set:** short chat prompts and long-context prompts.
- **Metrics:** perplexity, exact-match on needle-in-haystack retrieval, token-level divergence, and estimated KV memory.
- **Ablations:** no residual, no retention, no eviction, no transform, fixed-bit vs adaptive-bit.

Expected result profile:

- TurboQuant baseline: quality-neutral around 3.5 bits and strong 3-bit performance.
- TurboQuant-H: lower effective memory due to selective retention + eviction, with better quality than naive sub-2-bit quantization.

## 8. Limitations

TurboQuant-H is a research proposal, not a verified replacement for TurboQuant. The 20–30× target is realistic only when the task tolerates aggressive cache eviction or summarization. For exact long-context fidelity, a lower compression target may be more appropriate.

## 9. Conclusion

TurboQuant is a principled, data-oblivious quantization method with strong theoretical grounding and impressive practical results. TurboQuant-H preserves its strengths but adds structure-aware memory control so the system can move beyond uniform low-bit quantization and into the regime where 20–30× effective KV memory reduction becomes plausible.

## References

- Google Research Blog, *TurboQuant: Redefining AI efficiency with extreme compression*, 2026.
- Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*, arXiv:2504.19874.
- Hugging Face model card for `HuggingFaceTB/SmolLM-135M-Instruct`.
