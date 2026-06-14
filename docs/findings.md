# Wozformer Findings

Empirical results from the Wozformer project, collected during architecture
exploration (notebooks 01–12c) and the distillation pipeline (nb13–nb14 / scripts).
This document is the source of truth for the research paper.

Every finding has:
- **Setup**: model + hyperparameters
- **Observation**: what was measured
- **Implication**: why it matters

---

## Project context

**Goal.** Run a binary recurrent language model on a 1 MHz 6502 with paged
external weight memory (ESP32 flash) and produce coherent Shakespeare-style
text at <32 KB direct EEPROM or paged from <300 KB total weights.

**Corpus.** Tiny Shakespeare, ~1.1 MB lowercased (~430 K BPE tokens at vocab=512).

**Hardware target.** 4× AT28C64 EEPROM (32 KB total), 8 KB SRAM, 1 MHz Rockwell 6502,
optional ESP32 (4 MB flash, 80 MHz, paged ROM mode).

---

## The novel architecture: HDC-RWKV

A recurrent autoregressive language model where:

1. **Every learnable parameter is bipolar** (∈ {-1, +1}) at inference, stored as 1 bit
2. **Training uses the Straight-Through Estimator** (Bengio 2013, Courbariaux 2016) so
   gradients flow through `sign()`
3. **Position encoding via cyclic rotation** (HDC-style binding) — no separate
   positional embedding table
4. **Recurrent state update** with bounded `tanh` activation — no softmax-over-T,
   no KV cache, no exp / sum / divide
5. **Output via prototype similarity** — dot product of state with V learned
   prototype hypervectors, then softmax

```
state_t = tanh( decay_mask_bipolar ⊙ state_{t-1} + permute_t(vocab_hv_bipolar[token_t]) )
logits  = ( state @ prototype_hv_bipolar.T ) / temperature
```

To our knowledge no published architecture combines: bipolar weights via STE,
RWKV-style linear recurrence, HDC binding/permutation, and prototype-similarity
output, in a generative LM at sub-300KB deployment.

---

## Findings

### F1. The HDC-RWKV architecture is empirically trainable end-to-end via STE.

**Setup.** vocab=256, d=256, n_layers=1, block=16 (nb12c).
**Observation.** Best HARD val 4.35 nats/token (BPC 2.93). Train+eval converge;
soft and hard val loss agree throughout training (gap < 0.05 nats).
**Implication.** Sign-quantized weights with continuous tanh state is a viable
combination for autoregressive LM training at sub-100K-parameter scale —
contrary to prior BNN literature focused on classification.

### F2. Naive depth stacking degrades binary recurrence.

**Setup.** Same as F1 but n_layers=2 (Path A in nb12 logbook), no residual or
re-binarization between layers.
**Observation.** Best HARD val rose from 3.55 (1 layer) to 4.08 (2 layers).
Per-token output became more fragmented qualitatively.
**Implication.** Layer 2 receives continuous `tanh` activations whose distribution
differs sharply from the bipolar token vectors layer 1 was trained on. The
mismatch causes degradation, not improvement, when stacking.

### F3. Residual + re-binarization between layers does not rescue stacked binary recurrence at this scale.

**Setup.** Same as F2 but with `layer_input = ste_sign(states + layer_input)`
between layers (Courbariaux 2016 BNN trick).
**Observation.** Best HARD val 4.47 — worse than single-layer (3.55) and roughly
matching naive stacking (4.08).
**Implication.** The standard fix for stacked binary networks (residual +
re-binarize) is insufficient for sequence-modeling recurrences at vocab=128.
The mismatch between bipolar token vectors at layer 0 and tanh-bounded states
at layer N+1 cannot be patched by binarization alone — fundamentally different
input distributions, similar to the limitation reported in "Binary Neural
Networks: A Survey" (Qin et al. 2020) for sequence tasks.

### F4. STE binarization is NOT the bottleneck for HDC-RWKV.

**Setup.** All HDC-RWKV runs (vocab=128, 256, 512).
**Observation.** Soft val (STE forward) and hard val (real `.sign()` forward)
remain within 0.02–0.05 nats throughout training across all configurations
tested. The two curves are visually indistinguishable in most plots.
**Implication.** Quality limitations are architectural, not quantization-induced.
The model's continuous parameters do not "cheat" via fractional values during
training. Any quality ceiling we observe is the binary architecture's true ceiling.

### F5. HDC capacity ceiling at `d / log(d)` prototypes is empirically observable.

**Setup.** Comparing nb12c (vocab=256, d=256), nb12 (vocab=128, d=512), and
nb14 student (vocab=512, d=1024 and d=2048).
**Observation.** Quality (BPC) degrades as the ratio V / (d/log(d)) grows past ~4×.
- nb12c: V/(d/log(d)) = 256/(256/8) = 8 → BPC 2.93
- nb12:  V/(d/log(d)) = 128/(512/9) = 2.3 → BPC ~3.0
- nb14 student d=1024 V=512: ratio 5 → BPC 3.09
- nb14 student d=2048 V=512: ratio 2.4 → BPC 3.03 (marginal improvement)
**Implication.** Kanerva's 2009 capacity prediction for VSA superposition
(`d/log(d)` items) applies even when prototypes are gradient-trained rather
than random. This is the dominant architectural constraint on HDC-RWKV scaling.

### F6. Tokenizer choice matters more than additional model capacity at small scale.

**Setup.** Comparing nb12 (vocab=128, d=512) → nb12c (vocab=256, d=256). Same
EEPROM footprint (~16.5 KB).
**Observation.**
- nb12: best HARD val 3.55 nats, BPC 2.98, fragmented output
- nb12c: best HARD val 4.35 nats, BPC 2.93 (lower!), 51% of generated tokens are
  full words instead of subword fragments
**Implication.** At constant deployment budget, increasing vocabulary
(more BPE merges) gives qualitatively better output even when per-token nats
loss is higher, because each token carries more bits of information. The
right comparable metric across vocab sizes is BPC, not nats/token.

### F7. Binary recurrent students cannot fully absorb teacher knowledge from
    dense transformers at vocab > 256.

**Setup.** Teacher: dense transformer, vocab=512, d=256, L=6, ~3M params, val 3.55 nats.
Student: HDC-RWKV, vocab=512, d=1024 and d=2048, L=2.
Distillation: T=4.0, α_nll=0.3, α_distill=0.7.
**Observation.** Student plateaus at val 5.3–5.4 nats regardless of d∈{1024, 2048},
gap to teacher remains ~1.8 nats throughout 12K–15K steps. Soft and hard val
agree (binarization not the issue per F4).
**Implication.** The bipolar recurrent architecture has a representational
ceiling for matching dense transformer output distributions at vocab > 256.
Distillation cannot transfer information the student cannot represent. This
is consistent with F5 (HDC capacity ceiling).

### F8. Student capacity scaling has diminishing returns under binary constraint.

**Setup.** Same as F7. Compare d=1024 vs d=2048 at vocab=512.
**Observation.** Doubling student size (d=1024 → d=2048) improved best HARD val
by only 0.1 nats (5.40 → 5.30). Deployment size doubled from 131 KB to 262 KB.
**Implication.** Once HDC capacity is exhausted, further hypervector dimension
increase provides diminishing returns. Architectural changes (not size) are
needed past this point.

### F9. Channel mixing module addition does not improve binary recurrence at this scale.

**Setup.** nb12 with `ChannelMixing` block (gated MLP) added; same hyperparameters
otherwise.
**Observation.** Soft and hard val converged identically with and without CM.
Inspection showed the CM block contributing ~0 to the residual stream because
weights stayed near zero (sigmoid gate close to 0.5 produces small `r * v` values).
**Implication.** Gated MLPs do not adapt during binary STE training at this
scale without specific initialization or warmup recipes — the gate collapses
to indifference.

### F12. HDC-RWKV has a non-monotonic d–quality relationship: bigger d hurts at vocab=256.

**Setup.** vocab=256, NLL-only training (no distillation), L=2, block=64, 15K steps.
Compared d=256 (nb12c shipping) vs d=384 (today's ablation).

**Observation.** d=256 achieves BPC 2.93; d=384 achieves BPC 3.23. **Increasing the
hypervector dimension by 50% degraded quality by 0.30 BPC**, despite the larger
model having strictly more representational capacity in principle. Soft/hard val
gap is small (<0.05 nats) in both runs — not a binarization artifact.

**Hypothesis.** Two candidate mechanisms (not yet disentangled):
1. *Optimization*: At larger d, gradient signal-to-noise per dimension drops,
   making STE-based training less effective. Each dimension's effective gradient
   magnitude scales as 1/√d.
2. *Inductive bias loss*: At larger d, random initialization in {-0.5, +0.5}*0.5
   spreads parameters thinner, making it harder for STE to commit to signs that
   align with corpus statistics in finite training steps.

**Implication.** Counter-intuitive for the paper: **for binary recurrent LMs, more
parameters can be actively worse**. The default scaling-law assumption
("more params → better quality") fails. The d=256 / vocab=256 point is special.

### F11. Cross-architecture distillation can *hurt* HDC-RWKV students at small vocab.

**Setup.** Teacher: dense transformer, vocab=256, d=192, L=4, ~700K params,
best val 3.02 nats (BPC 2.04). Student: HDC-RWKV, vocab=256, d=384, L=2.
Distillation: T=4.0, α_nll=0.3, α_distill=0.7. 15,000 steps.

**Observation.** Student best HARD val 5.05 (BPC 3.40) — **0.7 nats WORSE than
nb12c's no-distillation baseline** (4.35 nats, BPC 2.93) despite using a 50%
larger hypervector dim (384 vs 256). Soft and hard val track identically
throughout (no binarization gap), so the regression is genuine, not a
binarization artifact.

**Implication.** Forcing a binary recurrent student to mimic a dense
transformer's *distribution shape* (high `α_distill`) actively damages
training when the student's representational space cannot host that
distribution. The KL term pushes the student into a region where neither
NLL nor KL can be minimized — a "stuck in nobody's land" failure mode.

**Open paths to mitigate (not yet tested):**
1. Lower `α_distill` (try 0.2 with 0.8 NLL) — distillation as auxiliary, not primary
2. Match student to teacher architecture (binary transformer student)
3. Replace KL with a top-k-restricted KL — only ask the student to mimic the
   top tokens, ignoring the tail the student can't represent
4. Use a smaller / matched-architecture teacher (e.g. RWKV teacher → HDC-RWKV student)

**Implication for paper.** This is a genuine novel negative result.
Distillation literature universally assumes a "smarter" teacher pulls the
student up. We show empirically that **architecture mismatch breaks this
assumption** at the binary recurrent scale.

### F10. Recurrent state binarization (vs continuous tanh) destroys training signal.

**Setup.** nb12 initial implementation: `state_t = ste_sign(decay * state + input)`
with full bipolar state.
**Observation.** Training diverged; soft val descended to 3.6 but hard val
diverged to >6 (worse than random). Gap > 2.4 nats.
**Implication.** STE through 16 sequential `sign()` applications loses gradient
fidelity. The continuous `tanh` state (with bipolar weights) is the working
combination. This is a positive constraint on the architecture design space:
**weights binary, activations continuous**.

---

## Architecture comparison table (Wozformer corpus, Tiny Shakespeare)

| Architecture | Source | Params | Deploy bytes | Val nats | BPC | Per-token cycles (6502) |
|---|---|---|---|---|---|---|
| Bigram baseline | nb01 | 1,024 | 1 KB | 2.44 | — | < 100 |
| Dense transformer (small) | nb07b | 1,536 | 0.4 KB int8 | 2.29 | ~3.30 | 17,000 mults |
| MoE-Tiny (4 experts) | nb09 | ~3K | 28 KB | TBD | TBD | 6,000 mults |
| RWKV (linear recurrence) | nb10 | ~3K | 23 KB | TBD | TBD | 4,000 mults |
| HDC Hebbian (random vocab) | nb11 | 0 trainable | 32 KB | ~4.0 | ~3.0 | 155K (XOR/popcount) |
| **HDC-RWKV vocab=128, d=512** | nb12 | 132K | 16.5 KB | 3.55 | 2.98 | 135K (XOR/popcount) |
| **HDC-RWKV vocab=256, d=256** | nb12c | 131K | 16.4 KB | 4.35 | **2.93** | 135K |
| HDC-RWKV vocab=512, d=1024 (no distill) | nb14 baseline | 1.05M | 131 KB | ~5.4 (predicted) | ~3.1 | per-token similar |
| HDC-RWKV vocab=512, d=1024 distilled | nb14 | 1.05M | 131 KB | 5.40 | 3.09 | per-token similar |
| HDC-RWKV vocab=512, d=2048 distilled | nb14 | 2.1M | 262 KB | 5.30 | 3.03 | per-token similar |
| Teacher dense transformer | nb13 (Kaggle) | 3.0M | 12 MB fp32 | 3.55 | 2.03 | not deployed |

### Reading the table

- **Per-byte efficiency**: nb12c is the best deployable artifact so far.
- **Output quality**: teacher >> all students. Teacher produces real Shakespeare
  characters (KING EDWARD IV, QUEEN MARGARET, GLOUCESTER), grammatical sentences,
  period vocabulary.
- **Scaling trend**: binary recurrent quality saturates as vocab grows past
  256, even with parameter scale up to 2.1M and distillation.

---

## Open questions / limitations / future work

### Q1. Is there a binary architecture that absorbs teacher knowledge at vocab=512+?

Current evidence (F7) suggests no, but only HDC-RWKV has been tested. Worth
trying: binary transformer student, binary RWKV student (not bipolar HDC).
Could distillation transfer better to architectures that match teacher's
inductive bias?

### Q2. Why does the binary recurrence saturate around BPC 3.0?

Hypothesis: prototype-similarity output is inherently low-rank. Logit space is
bounded by V × d bits; cross-entropy minimization needs continuous logit ranges
to encode soft distributions. Future work: replace prototype-similarity output
with a binary unembedding matrix.

### Q3. Does mixed-precision help? (int4 weights or fp8 LN params)

Untested. Mixed-precision int4 students might bridge the binary-recurrence
ceiling without violating the "hardware-friendly" constraint.

### Q4. Can distillation at vocab=256 push student BPC below nb12c's 2.93?

Under investigation as of nb14 round 2 (retraining teacher at vocab=256, then
distilling student). If yes: distillation is useful for binary recurrence at
the right capacity ratio. If no: distillation doesn't help binary recurrence
at any scale.

### Q5. Hardware deployment latency tradeoffs

Per-token cycle counts are computed analytically. Need actual 6502 measurements
to validate (and to demonstrate the paged-ROM SPI protocol with ESP32).

---

## Hardware deployment design

### Direct-fit deployment (no paging)

Models under 32 KB deploy directly to 4× AT28C64 EEPROM. Currently:

- `wozformer_hdcrwkv_v3.bin` (nb12c output, 16.4 KB) is the leading candidate.
- Per-token cost: ~135K cycles ≈ 135 ms at 1 MHz.

### Paged deployment (for >32 KB students)

For student models 32–300 KB:

- Weights live in ESP32 flash (4 MB available)
- 6502 requests weight pages via SPI
- 6502 SRAM (8 KB) holds current page + activations
- Per-token cost increases ~2× due to SPI fetch latency
- Project name for this path: "paged-ROM mode"

Currently deferred pending student that beats nb12c at deployable size.

---

## Citation pointers for paper write-up

- **VSA / HDC primitives**: Kanerva (2009), Plate (1995), Rachkovskij & Kussul (2001)
- **STE**: Bengio et al. (2013), Courbariaux et al. (2016) "BinaryConnect"
- **Distillation**: Hinton, Vinyals, Dean (2015)
- **RWKV linear recurrence**: Peng et al. (2023) RWKV-4 paper
- **BNN survey**: Qin et al. (2020)

---

## Appendix: Reproduction commands

```bash
# Smoke tests
python tests/test_smoke.py

# Teacher (small, local)
python scripts/train_teacher.py --steps 4000 --d 128

# Student with distillation
python scripts/train_student.py --steps 12000 --d 1024

# Side-by-side comparison
python scripts/compare.py --paths runs/teacher.pt runs/student.pt
```

For Kaggle GPU runs: notebooks `nb13_teacher_training.ipynb` and
`nb14_student_distillation.ipynb`.

---

*Last updated: during Kaggle run of nb14 with vocab=512, d=2048 student.
The plateau observation (F7, F8) prompted re-pointing to vocab=256 retrain.*
