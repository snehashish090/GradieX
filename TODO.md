# TODO — performance

Scope: making v2 faster. Feature gaps (save/load, dropout, L2, softmax backward,
validation split, seeding) are tracked separately — but see note in step 7, because
mini-batching is both a feature and the single largest performance lever.

Every figure below was measured on this machine (Apple M5, 4P+6E, Apple clang 21,
`-O2`, one thread unless stated, shape `128 -> W x2 -> 10`, batch 1, float64) unless
labelled **[est.]**. Payoffs labelled **[measured]** were verified against the real
engine or a prototype of the exact change.

Current state: `core/layer.h` and `core/network.h` are the unmodified baseline.
None of the steps below are applied.

---

## Reference: where the ceilings actually are

| | GFLOP/s, 1 core | note |
|---|---|---|
| v2 today, float64 | ~11.5 | measured flat across W=32..2048 |
| NEON fp32 FMA peak | 45.3 | measured, `vfmaq_f32` back to back |
| NEON bf16 matmul (`BFMMLA`) | 141.2 | measured, 3.1x fp32 |
| NEON int8 matmul (`SMMLA`) | 188.5 (GOP/s) | measured, 4.2x fp32 |
| Accelerate / AMX, fp32 | 1006 | PyTorch's backend, ~1.5 cores |

Two facts that bound everything:

1. **fp32 has almost no headroom left.** A well-tuned register-tiled fp32 kernel
   reached 129 GFLOP/s on 4 threads = 32/core, which is 71% of the 45.3 peak.
   Tuning fp32 further is not worth the effort.
2. **The matrix unit is closed on macOS.** `FEAT_SME`/`FEAT_SME2p1` are reported by
   sysctl, but any SME or SVE instruction dies with SIGILL — the OS does not let user
   processes enter streaming mode. `FEAT_BF16` and `FEAT_I8MM` *do* work (step 9), and
   Linux does permit the matrix units (step 10).

---

## Step 0 — Benchmark harness and a perf regression test

**Do this first.** The width-512 regression below was invisible for the entire life of
v2 because nothing measured it. Nothing after this step can be verified without it.

- Add `bench/` with a sweep over `W = {32, 64, 128, 256, 512, 1024, 2048}`, fixed
  shape `128 -> W x2 -> 10`, reporting **nanoseconds per multiply-accumulate**
  (time / parameter count), not raw microseconds. Normalised units are what make a
  scaling break visible — flat means correct scaling, any climb is a real defect.
- Adaptive iteration count targeting a fixed wall-clock budget (~0.25 s) per point,
  best-of-3.
- Add one CI check: `ns/MAC` at W=1024 must not exceed the W=128 figure by more than
  ~25%. That single assertion would have caught the step-2 bug immediately.

**Measurement discipline — this bit caused real confusion, don't skip it:**

- This machine runs roughly **2x slower warm than cold**, and the shift applies to
  C++ but *not* to PyTorch (whose batch-1 path is dispatch-bound). The same v2 binary
  measured 100.78 us cold and 195.85 us warm at W=512.
- Therefore: never compare a number taken now against a number taken in another
  session, and re-measure every engine back to back inside one run whenever the
  comparison matters.
- Power-of-two widths are pathologically slow with strided access (cache-set
  aliasing). Keep both power-of-two and non-power-of-two widths in the sweep —
  W=2048 vs W=2896 differed by 2x for this reason.

---

## Step 1 — Row-major backward loop order

**`core/layer.h:167`, `HiddenLayer::computeErrorSignal`.**
**[measured] 2.1x at W=512. This is the bug, not an optimisation.**

Currently the neuron index is the outer loop, so
`nextLayerWeights[nextNeuron * layerWidth + neuron]` strides `layerWidth * 8` bytes per
read — a 4 KB stride over a 2 MB matrix at W=512, re-walked once per neuron, fetching
one cache line per useful double.

Swap the loops: accumulate `W^T * delta` with `nextNeuron` outer and `neuron` inner so
the weight matrix is read row-major and contiguously. Accumulate into a
`layerWidth`-sized buffer, then apply the activation derivative.

While in here, fix the second problem in the same function (`core/layer.h:182`): the
derivative is invoked through the function pointer **once per neuron with `size == 1`**.
That is 1536 indirect, un-inlinable, un-vectorisable calls per update at W=512x3, and it
defeats the entire point of the vectorised signature. Hoist it to one call per layer.

Consequence: this restores textbook quadratic scaling. Baseline grows 4.4x then 6.5x
per width doubling (super-quadratic); fixed it grows 3.8x/3.9x, i.e. ~4x as it should.
Without this fix v2 is **slower than GradieX v1** from W=512 up (3.0x slower at W=4096).

## Step 2 — FP reassociation on the dot product

**`core/layer.h:99`, inner loop of `calculateOutputs`.**
**[measured] 4.15x on the isolated kernel (94.4 -> 23.0 us at 512x512).**

`sum += row[i] * x[i]` is a serial chain of N floating-point adds. The compiler may not
reassociate it without permission, so it cannot split the reduction across vector lanes,
and the loop runs at 5.6 of ~25 available GFLOP/s.

```cpp
double sum = 0.0;
const double* __restrict__ row = this->neuronWeights.data() + neuron * this->inputSize;
{
    #pragma clang fp reassociate(on)
    for (int i = 0; i < this->inputSize; ++i) sum += row[i] * input[i];
}
```

Use the **scoped pragma, not global `-ffast-math`** — it relaxes associativity for this
one loop while leaving NaN/Inf semantics and the `eps` clamps in `core/loss.h` intact.
Do **not** hand-unroll into 4 accumulators as well; that was measured *slower* (40.8 vs
22.8 us) than letting the compiler vectorise.

**Order matters: land this before writing the numerical gradient check**, since it makes
results compiler- and flag-dependent in the last bits.

## Step 3 — Fused gradient apply (batch-1 fast path)

**`core/layer.h:110`/`122` and `core/network.h:127`/`129`.**
**[measured] ~1.2x.**

`computeGradients` materialises the whole `width * inputSize` gradient matrix, then
`updateWeightsAndBiases` reads it straight back — a 2 MB write plus 2 MB read per layer
per update at W=512, for a batch of one. Add an `applyGradientsFused(lr)` that writes
`delta * input` directly to the weights and skips the buffer.

**Two constraints:**
- Only valid once **every** layer's delta is computed. Applying per-layer while walking
  backward corrupts every layer behind it, because a hidden layer's error signal reads
  the *next* layer's weights before they move.
- Only valid at batch 1. Keep `computeGradients` and the gradient buffer — step 7 needs
  them back. This is an added fast path, not a replacement.

## Step 4 — Scratchpad sizing

**`core/layer.h:65-66`.**
**[measured] peak RSS 736 MB -> 370 MB at W=4096. Speed-neutral. Also turns a failing
test green.**

```cpp
this->scratchPad1.resize(width);   // was inputSize*width
this->scratchPad2.resize(width);
```

Both pads only ever hold one value per neuron (a loss derivative and an activation
derivative). `inputSize*width` overallocates by a factor of `inputSize` — 268 MB of
waste per layer at W=4096 — and is simultaneously too **short** when `inputSize == 0`.
`tests/test_layer.cpp` already documents the second half of this bug in
`Construction_ScratchPadsHoldAtLeastOneValuePerNeuron`.

### Checkpoint after steps 1-4

**[measured]** combined effect, batch 1, float64:

| W | before | after | gain |
|---|---|---|---|
| 512 | 908.95 us | 195.63 us | 4.65x |
| 1024 | 3866.82 | 734.04 | 5.27x |
| 4096 | 119621 | 12249 | 9.77x |

ns/MAC goes from 0.89->6.89 (climbing, broken) to 0.41->0.71 (near-flat across a
3000x range in parameter count). Test suite unchanged except the one fix from step 4.

---

## Step 5 — Row blocking in the forward pass

**`core/layer.h:82`, `calculateOutputs`.**
**[measured] 1.22-1.33x.**

After steps 1-4 the engine streams weights at ~46 GB/s. One core demonstrably does
**116.7 GB/s** on this exact problem (verified: TorchScript, cpu/wall = 1.00), so you
are at 39% of what the core can do. One accumulator chain means too few outstanding
loads to saturate.

Compute **K output neurons per pass** over the input — K independent accumulator chains,
K weight streams in flight, and the input vector reused from registers:

```cpp
for (int n = 0; n + K <= out; n += K) {
    const double* r0 = W + (n+0)*in; /* ... r1, r2, r3 */
    double s0=0, s1=0, s2=0, s3=0;
    for (int i = 0; i < in; ++i) {
        const double xi = x[i];
        s0 += r0[i]*xi; s1 += r1[i]*xi; s2 += r2[i]*xi; s3 += r3[i]*xi;
    }
    /* + bias, store */
}
```

**Use K=2 or K=4.** Measured best: K=4 at W=128 (59.7 GB/s), K=2 at W=512 and W=2048.
K=8 and K=16 both regress — too many live weight streams for the register file. Apply
the same blocking to the step-1 backward loop.

## Step 6 — float32 support

**[measured] 1.6-2.9x** (0.937 -> 0.532 us at W=32; 6.044 -> 2.115 at W=128;
58.159 -> 36.010 at W=512).

Template the engine on the scalar type. This halves bytes per weight, which is the
actual constraint, and doubles NEON lanes per instruction. It is the **largest single
lever that helps batch-1 latency** — steps 7 and 8 do nothing for latency.

Keep float64 available and keep the gradient check on the float64 path.

---

## Step 7 — Mini-batching (GEMV -> GEMM)

**[measured] ~13-17x on bulk training throughput** at W=2048 (2999 us/sample ->
~180-240 us/sample). Does **nothing** for batch-1 latency.

A batch-1 matrix-vector product reads every weight once and does 2 flops with it —
arithmetic intensity 0.25 flop/byte in f64. No code change improves that. Batching
raises intensity ~B-fold: read a weight row once, use it for B samples.

**This is also the missing `TrainOptions` feature**, so it is one piece of work with two
payoffs. It needs the gradient accumulation from step 3 kept intact.

Three things that matter, learned the hard way:

1. **Store activations feature-major** — `At[feature * B + batch]` — so the batch is the
   contiguous inner dimension and every inner loop is a long vectorisable run.
   Sample-major kills it.
2. **Hold the accumulator tile in registers**, not memory. My first attempt accumulated
   into `C[n][j]` in memory and got 18.2 GFLOP/s; a `KN x KB` register tile got **55.5
   single-thread / 129.1 on 4 threads**. Same arithmetic, 3x apart. This is the single
   biggest implementation detail.
3. **Tile 4x16.** Measured: 4x8 -> 79.7, **4x16 -> 129.1**, 6x16 -> 115.8, 8x16 -> 82.5
   (GFLOP/s, 4 threads). Larger tiles spill registers.

Also replace `std::vector<std::vector<double>>` in `core/network.h:154` with a flat
buffer plus a shuffled index array — one heap allocation per sample is ~360 MB of
scattered memory for 60k x 784, and a flat layout is a prerequisite for batching anyway.

## Step 8 — Multithreading

**[measured] 2.5x from 1 -> 4 threads** in the batched prototype (1268 -> 498 us/sample).

Only worth doing **after** step 7, and only for batched work. At batch 1 thread wake and
sync cost 5-20 us while an entire W=128 forward pass is 6 us, so it is pure loss there.

Parallelise over output neurons for forward and `dW`; parallelise over the *input* range
for `dA` to avoid write races. At B=256 a batch step is tens of milliseconds, so
spawning threads per region costs ~2% — a persistent pool is a refinement, not a
requirement.

---

## Step 9 — bf16 / int8 matrix instructions (NEON, no library)

**[measured] BFMMLA 141.2 GFLOP/s vs 45.3 for fp32 FMA = 3.1x. Works on this Mac today.**

`FEAT_BF16` and `FEAT_I8MM` are reported and functional — these are ordinary Advanced
SIMD instructions, **not** SME, so no streaming mode and no library:

- `vbfmmlaq_f32(float32x4_t, bfloat16x8_t, bfloat16x8_t)` — 2x4 by 4x2 bf16 product
  accumulating into a 2x2 fp32 tile, 32 flops per instruction vs 8 for `vfmaq_f32`.
- `vmmlaq_s32` — int8, 188.5 GOP/s, for quantised inference.

Build with `-march=armv8.6-a+bf16+i8mm`. On 4 cores this projects to **~565 GFLOP/s
[est.]**, within ~1.8x of Accelerate's 1006 — without linking anything.

bf16 inputs with fp32 accumulation is exactly what mixed-precision training does.

**Numerics caveat — do not let this quietly invalidate the gradient check.** bf16 has
8 mantissa bits; verifying to 1e-11 is impossible in it. Keep a master copy of the
weights in fp32, use bf16 only for matmul inputs, accumulate in fp32, and keep the
gradient check on the fp64 path.

## Step 10 — Linux, for the matrix units

macOS will not give you SME. Linux will give you the equivalent.

**x86-64 with Intel AMX** (Sapphire Rapids / 4th gen Xeon or later) is the best target.
AMX tile instructions are documented and reachable through `<immintrin.h>` with no
library — the only requirement is one syscall at startup:

```c
syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
```

`TDPBF16PS` / `TDPBSSD` then give bf16 and int8 matrix multiply on the order of
1-2 TFLOP/s per core **[est., spec-derived, unmeasured]** — roughly 8-16x AVX-512 fp32.
This is the path to genuinely matching vendor-BLAS throughput with zero dependencies,
which is precisely what Apple denies you.

**ARM Linux** (Graviton3/4, Grace) gives SVE/SVE2 plus the same BF16/I8MM ops, and
permits SME on hardware that has it. Check `/proc/cpuinfo` for `sve2 bf16 i8mm sme`
rather than assuming.

---

## Do not bother — measured to give nothing

Saving these so the effort isn't spent twice. All were implemented and timed:

| change | result |
|---|---|
| Fuse activation into the neuron loop | 1.02-1.04x — noise |
| Keep input by pointer instead of copying | included above — noise |
| Compile-time layer sizes (full unroll) | **0.87-0.99x — slower.** Over-unrolling thrashes the instruction cache at W=512 |
| Hand-unroll the dot product into 4 accumulators *with* the pragma | 40.8 vs 22.8 us — **slower** than letting the compiler do it |
| Threading at batch 1 | sync cost exceeds the whole forward pass |
| SME / SVE intrinsics on macOS | SIGILL — OS-gated, unavailable |

The pattern: reducing **instruction count** does nothing, because this engine is not
compute-bound. Increasing **register-level reuse** and **memory-level parallelism**
(steps 5, 7) and **reducing bytes per weight** (steps 6, 9) are what work.

---

## Suggested order

1. Step 0 — harness first, or nothing else is verifiable
2. Steps 1-4 — measured 4.6-9.8x, small and local, no API change
3. Steps 5-6 — measured ~2-4x combined, still local; best latency-per-effort
4. Step 7 — the big structural one, doubles as a missing feature
5. Steps 8-9 — throughput and data-type work
6. Step 10 — only if chasing vendor-BLAS parity is actually a goal

## Where this can realistically land

- **Batch-1 latency, small models:** already 8-54x faster than PyTorch at W<=128.
  Steps 5-6 and 9 extend that. This is the defensible claim.
- **Batch-1 latency, large models:** roughly parity with PyTorch from W~900. Steps 5-6
  put you clearly ahead.
- **Batched throughput:** steps 7-8 give 13-17x over today and land ~5x behind PyTorch.
  Step 9 closes that to ~2x. Matching it on macOS requires Accelerate.
- **Do not chase "faster than PyTorch" in fp32 on this hardware.** Accelerate runs
  5.6x above the entire 4-core NEON fp32 peak via a coprocessor the OS won't expose.
  "Within 2x of vendor BLAS with zero dependencies" is both achievable and a stronger
  claim than one you'd have to defend.
