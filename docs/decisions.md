# Design Decision Log — Radar Pulse Classification

This file records the technical and strategic decisions taken over the course of the project.
It is a **chronological, append-only record**: superseded entries are kept and annotated rather
than deleted, so the reasoning behind the final design remains auditable.

Each entry follows the structure:

```
## YYYY-MM-DD — [Decision Title]
- **Decision:** what was done / will be done
- **Rationale:** why this option
- **Alternatives:** what else was considered, and why it was rejected
- **Impact:** what this decision affects
```

> This log is the primary source for the manuscript's *Methods* and *Implementation Details*
> sections. Keep it detailed.

---

## 2026-05-03 — Research Topic and Academic Framing

- **Decision:** Deep-learning classification of radar pulse signals under low-SNR conditions,
  comparing spectrogram (STFT), Choi–Williams (CWD), and Wigner–Ville (WVD) representations.
- **Rationale:** Most of the literature reports high accuracy at high SNR, whereas real
  electronic-warfare environments are dominated by low-SNR conditions. A noise-robustness
  analysis raises the work to a publishable contribution.
- **Alternatives:** Autonomous swarm simulation, ballistic trajectory optimisation,
  image-based target detection, strategic analysis from satellite imagery, encrypted tactical
  messaging.
- **Impact:** Every module choice and the whole technology stack follows from this decision.

---

## 2026-05-03 — Number of Target Classes: 8

- **Decision:** Eight radar signal classes (LFM, NLFM, Barker, Frank, polyphase P1–P4, Costas,
  CW, Stepped/FH).
- **Rationale:** The standard range in the academic literature is 6–12 classes. Eight provides
  sufficient diversity while remaining trainable in reasonable time on the available hardware.
- **Alternatives:** 4–5 classes (insufficient diversity), 12+ classes (training time and VRAM
  become difficult).
- **Impact:** Scope of the data-generation module, model output dimensions, confusion-matrix size.

---

## 2026-05-03 — SNR Range: −10 dB to +20 dB

- **Decision:** SNR range of −10 dB to +20 dB for both training and test (2 dB steps).
- **Rationale:** In real EW environments the lower bound is around −10 dB; +20 dB represents
  clean-signal behaviour. This range is the most common reference in the literature.
- **Alternatives:** Starting at −20 dB (too extreme, meaningless results), starting at 0 dB
  (unrealistic).
- **Impact:** Basis of the data-generation parameters; the x-axis of every SNR robustness curve.

---

## 2026-05-03 — Development Environment: Hybrid (Local + Cloud)

- **Decision:** Local machine for code development and MATLAB data generation; Kaggle for ResNet
  training; Colab Pro for ViT/Swin training.
- **Rationale:** The RTX 5050 Laptop GPU (8 GB VRAM) is adequate for prototyping but constrained
  for large models. Kaggle offers a free T4 and Colab Pro gives cheap A100 access. AWS was
  deliberately excluded (hourly cost, academic budget).
- **Alternatives:** Fully local (slow, ViT untrainable), fully AWS (expensive), a single cloud
  provider (inflexible).
- **Impact:** Data format (must stay small enough to upload to Kaggle/Colab), config management
  (cloud/local separation).

> **Outcome (2026-05-25):** In practice all nine experiments ran locally on the RTX 5050. The
> decimated CWD/WVD implementations were fast enough that the cloud fallback was never needed.

---

## 2026-05-03 — Python 3.11.9 (instead of 3.14)

- **Decision:** The project runs on a Python 3.11.9 virtual environment.
- **Rationale:** Python 3.14.3 was present on the system, but critical libraries (PyTorch,
  `tftb`) had no official 3.14 wheels. 3.11 is the sweet spot: fully supported by every
  scientific library in the stack.
- **Alternatives:** Python 3.10 (older), 3.12 (mostly supported but less battle-tested than
  3.11), 3.14 (incompatibility risk).
- **Impact:** `requirements.txt` is pinned against 3.11.

---

## 2026-05-03 — Deep-Learning Framework: PyTorch (instead of TensorFlow)

- **Decision:** PyTorch + CUDA.
- **Rationale:** PyTorch dominates academic research; `timm` gives easy access to modern
  architectures such as ViT/Swin; the `grad-cam` package is mature for PyTorch.
- **Alternatives:** TensorFlow/Keras (good for deployment, but the research community has moved
  to PyTorch), JAX (not yet mature enough here).
- **Impact:** All model implementations use PyTorch; ONNX export remains available if deployment
  is ever needed.

---

## 2026-05-03 — License: MIT

- **Decision:** MIT License.
- **Rationale:** The de facto standard in academia; the most permissive terms; no barrier to
  others reusing the code.
- **Alternatives:** Apache 2.0 (adds patent protection, but nothing here is patentable), GPL
  (copyleft restrictions, rarely preferred in academia).
- **Impact:** Companies may also use the code; a citation requirement is stated in the README.

---

## 2026-05-04 — Sample Rate: 100 MHz

- **Decision:** Synthetic radar pulses are generated at a 100 MHz sample rate (Ts = 10 ns).
- **Rationale:** The de facto standard in the literature; provides ample margin over Nyquist for
  our chirp bandwidth range (5–20 MHz). A 200 MHz rate would double the signal vector, slow the
  O(N²) WVD/CWD computation by ~4×, and constrain batch size.
- **Alternatives:** 200 MHz (meaningful for high-bandwidth LFM but unnecessary in our scenario),
  50 MHz (adequate for narrowband signals but leaves a thin LFM bandwidth margin).
- **Impact:** `fs = 100e6` in every MATLAB generator. TF representation sizes and FFT lengths are
  built on this basis.

---

## 2026-05-04 — Pulse Width 1–20 µs, Fixed Signal Length 2048 Samples

- **Decision:** Pulse width drawn uniformly from [1 µs, 20 µs]. All signals are zero-padded to a
  fixed length of **2048 samples** (≈20.48 µs at 100 MHz).
- **Rationale:** 2048 = 2^11 is FFT-friendly. The 1–20 µs range covers typical EW scenarios
  (search radar 1–100 µs, tracking 0.1–1 µs, pulse compression 10–50 µs). Fixed length is
  required for batch processing; the padding position is also randomised (data augmentation).
- **Alternatives:** Variable length + RNN (complex), 1024 samples (long pulses would be
  truncated), 4096 samples (unnecessarily large).
- **Impact:** Generators draw `pulse_width = rand_in([1e-6, 20e-6])` and pad to 2048.

---

## 2026-05-04 — 5000 Samples per Class, SNR Assigned Randomly

- **Decision:** 5000 samples per class (40,000 total). Each sample is assigned one SNR value
  drawn from the `[-10, -8, ..., +20] dB` grid.
- **Rationale:** 5000 samples per class is trainable on the available hardware and fits within
  cloud dataset limits. Assigning SNR per sample gives generalisation equivalent to
  "generate every sample at every SNR" with far less data (continuous-augmentation logic). It
  still guarantees enough samples per (class, SNR) cell for SNR-stratified analysis
  (≈ 5000/16 ≈ 312 per class per SNR).
- **Alternatives:** 10,000 per class (80,000 total → excessive size), every sample at every SNR
  (16× data explosion).
- **Impact:** Two-layer generation loop: (1) generate 5000 clean signals per class, (2) assign an
  SNR to each.

---

## 2026-05-04 — SNR Step: 2 dB (16 Points)

- **Decision:** SNR grid `[-10, -8, -6, ..., +18, +20]` dB → 16 discrete points.
- **Rationale:** 2 dB is the de facto standard in the literature. A 1 dB step gives 31 points and
  a smoother robustness curve, but doubles the compute and data-management cost for marginal
  information gain.
- **Alternatives:** 1 dB (unnecessarily fine), 5 dB (curve too coarse), 3 dB (inconsistent with
  the literature).
- **Impact:** These 16 points form the x-axis of every SNR robustness curve.

---

## 2026-05-04 — Train/Val/Test Split: 70/15/15

- **Decision:** 70% train / 15% validation / 15% test, stratified by both class and SNR.
- **Rationale:** For 40,000 samples this gives 28k/6k/6k. A 6000-sample validation set makes
  early stopping and hyperparameter tuning safe. A 6000-sample test set gives ~750 samples per
  class — enough for a reliable confusion matrix.
- **Alternatives:** 80/10/10 (validation drops to 4000, risky for ViT), 60/20/20 (a smaller
  training set hurts large models), k-fold CV (academically attractive, but 9 experiments ×
  k folds is impractical).
- **Impact:** The split is performed in Python (for flexibility), not in MATLAB.
  See the 2026-05-22 entry for the final implementation.

---

## 2026-05-04 — Class Balance: Perfectly Balanced

- **Decision:** Exactly 5000 samples per class.
- **Rationale:** A controlled experimental setting is required for academic comparison. Class
  imbalance is a separate research question (re-sampling, focal loss, class-weighted CE) and
  would dilute the paper. Reviewers expect balanced classes as a baseline.
- **Alternatives:** Realistic imbalance (e.g. CW more frequent) — closer to operational data but
  complicates the academic baseline.
- **Impact:** Standard unweighted cross-entropy loss. The confusion matrix is directly
  interpretable.

---

## 2026-05-04 — File Format: HDF5 (.h5)

- **Decision:** The synthetic data is stored in a single HDF5 file. MATLAB writes it via
  `save('-v7.3')` (which is HDF5); Python reads it with `h5py`.
- **Rationale:** Hierarchical structure (signals/labels/snr/metadata in one file), chunking and
  compression support, and MATLAB v7.3 is already HDF5 — the two systems bridge naturally.
- **Alternatives:** `.npy/.npz` (metadata management is awkward), `.mat` v7 (2 GB limit),
  `.parquet` (tabular, unnatural for signals), `.tfrecord` (TensorFlow-specific).
- **Impact:** See `docs/dataset.md` for the actual on-disk layout, including the MATLAB
  column-major storage convention.

---

## 2026-05-04 — Random Seed: Global 42, Layered

- **Decision:** Master seed = 42, with per-module sub-seed management:
  - MATLAB generation: `rng(42, 'twister')`
  - NumPy: `np.random.seed(42)`
  - PyTorch: `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`
  - cuDNN: `deterministic = True`, `benchmark = False`
  - DataLoader: `worker_init_fn` gives each worker `seed + worker_id`
- **Rationale:** Full reproducibility is critical for an academic paper. If a reviewer runs the
  code and the numbers do not match, that is a problem. The layered structure prevents one
  module's seed from leaking into another.
- **Alternatives:** Random seed (no reproducibility), a single global seed (cross-module leakage),
  no seed (unacceptable).
- **Impact:** cuDNN deterministic mode costs ~5–10% speed; an acceptable trade-off.

---

## 2026-05-04 — NLFM Variants: Quadratic + Sinusoidal Mixture

- **Decision:** The NLFM class is not a single mathematical form; each sample randomly draws one
  of two variants:
  - **Quadratic NLFM** (60%): `f(t) = f0 + k1·t + k2·t²`
  - **Sinusoidal NLFM** (40%): `f(t) = fc + (B/2)·sin(2πt/T)`
- **Rationale:** A single variant risks overfitting the model to one mathematical form. The NLFM
  literature contains many sub-variants; two distinct curve types give enough diversity for the
  network to learn the "non-linear FM family" pattern. Quadratic is the most common baseline;
  sinusoidal looks visually very different in the TF plane (S-curve vs parabola). The mixture is
  also closer to real EW conditions, where different emitters use different NLFM shapes.
- **Alternatives:** Quadratic only (simple but monolithic), 4+ variants (unnecessary complexity
  at this stage), splitting variants into separate classes (breaks the 8-class target).
- **Impact:** `generate_nlfm.m` picks a variant at random and records it in `params.variant`.

---

## 2026-05-04 — Barker Codes: B7 + B11 + B13, Rectangular Chip

- **Decision:** Each Barker sample randomly selects B7, B11, or B13 (equal probability). Chip
  shape is rectangular (unfiltered, instantaneous phase jumps). Chip duration `Tc = T/N`.
- **Rationale:** B7/B11/B13 is the canonical set known as "the Barker codes"; B2–B5 are very
  short and rare in real systems. Equal probability prevents the model from overfitting to a
  single code length. A rectangular chip is the academic baseline and keeps the classes fair
  (no transmitter filter is modelled for LFM/NLFM either).
- **Alternatives:** B13 only (common in the literature but single-point), all Barker codes B2–B13
  (academically complete but B2–B5 are unrealistic), raised-cosine or Gaussian filtered chips
  (realistic but invites "which filter, and why?" questions).
- **Impact:** `generate_barker.m` records the chosen code in `params.code_name`.

---

## 2026-05-04 — Frank Polyphase: N ∈ {4,6,8}, fc = 0, Extended Pulse Width

- **Decision:** Three parameters for the Frank polyphase class:
  - **Matrix size N**: uniformly random from {4, 6, 8} (total chips N² → 16/36/64)
  - **Carrier frequency**: `fc = 0` (pure complex baseband)
  - **Pulse width**: a Frank-specific [4, 20] µs range
- **Rationale:** Mixing N is consistent with the 3-code approach used for Barker and prevents
  overfitting to a single matrix size. N = 5, 7 (prime) are rare in the literature. `fc = 0`:
  Frank's characteristic TF signature comes from its phase matrix; adding a carrier would mask
  it. The extended pulse width (T ≥ 4 µs) is required because at N = 8 (64 chips) and T = 1 µs,
  `Tc` would be 15.6 ns — only 1.56 samples per chip at 100 MHz (undersampled). A 4 µs lower
  bound guarantees at least 6.25 samples per chip in the worst case.
- **Alternatives:** Fixed N = 8 (most common baseline but monolithic), N ∈ {4,5,6,7,8} (primes
  are rare for Frank), random fc (signature masking), [1, 20] µs (chip undersampling).
- **Impact:** A dedicated pulse-width config field (later shared with the whole polyphase family).

---

## 2026-05-04 — P1–P4 Polyphase: All Sub-Codes, fc = 0, Shared Pulse Width

- **Decision:** The polyphase class draws one of four sub-codes with equal probability (25% each):
  P1 and P2 (Lewis–Kretschmer), P3 and P4 (LFM approximations). Matrix size N ∈ {4, 6, 8} as for
  Frank, `Nc = N²`, carrier `fc = 0`, pulse width shared with Frank ([4, 20] µs).
- **Rationale:** All four sub-codes are required for academic completeness — it closes the
  "why is P2 missing?" question. P3 and P4 are effectively **discretised LFM** ("stepped
  approximation of a chirp"), which makes the P3/P4-vs-LFM distinction genuinely hard and
  strengthens the paper's fine-grained-classification story. P1 and P2 are close to Frank
  (matrix-based discrete phases), creating an intra-family discrimination challenge.
- **Alternatives:** P3+P4 only (simple but incomplete), P1+P3+P4 (skips P2, hurts completeness),
  separate classes per P-code (breaks the 8-class target), random fc (inconsistent with Frank).
- **Impact:** `generate_polyphase.m` records `params.subcode` (P1/P2/P3/P4). The 8-class target is
  preserved.

---

## 2026-05-04 — Costas Frequency Hopping: N ∈ {5,6,7,8}, Symbolic Sequences, Symmetric Baseband

- **Decision:** Five parameters for the Costas class:
  - **N (code length)**: uniformly random from {5, 6, 7, 8}
  - **Costas sequence**: two canonical sequences predefined per N, chosen at random (8 total)
  - **Frequency step Δf**: uniform in [2, 5] MHz
  - **Pulse width**: the generic [1, 20] µs range
  - **Frequency placement**: symmetric baseband, centred on 0
- **Rationale:** N ∈ {5,6,7,8} is the standard Costas range (N = 3, 4 are too short; N ≥ 9 adds
  variance without benefit). Two sequences per N give enough diversity — using every known Costas
  sequence (e.g. 200 for N = 7) is unnecessary. The Δf range keeps the worst case (N = 8, 40 MHz
  total bandwidth) inside the guard band at 100 MHz. Symmetric baseband is consistent with the
  `fc = 0` choice for Frank/polyphase.
- **Alternatives:** Fixed N (monolithic), all known Costas sequences (excessive), Welch–Costas
  auto-generation (restricts N), fixed Δf (less diversity), asymmetric placement (no symmetry).
- **Impact:** `generate_costas.m` records N, the sequence, Δf, and the resulting frequencies. In
  the TF plane, Costas appears as N short blocks scattered across time–frequency — its signature.

---

## 2026-05-04 — CW (Continuous Wave): Generic Pulse Width, Random fc, Random Initial Phase

- **Decision:** Three parameters: generic [1, 20] µs pulse width; carrier `fc` uniform in
  `[-fmax, +fmax]` with a 5% guard band; initial phase uniform in `[0, 2π)`. Signal model:
  `s(t) = exp(j(2π·fc·t + φ0))`.
- **Rationale:** The generic pulse width prevents CW from being separable by duration alone (fair
  comparison). Random fc: Frank/polyphase use `fc = 0`, so a CW at DC could be confused with the
  first row of an N = 4 Frank code. A wide fc range forces the model to learn the "single
  horizontal line" pattern in a frequency-independent way. Random φ0: real transmitters are not
  coherent, and randomising phase prevents overfitting to padding/phase cues.
- **Alternatives:** [10, 20] µs (creates a pulse-width cue), fixed `fc = 0` (confusable with
  Frank), fixed φ0 (deterministic but unrealistic).
- **Impact:** TF signature: a single bright horizontal line at constant frequency. At −10 dB this
  single line is buried in noise.

---

## 2026-05-04 — Stepped/FH: Costas-Consistent Parameters, Monotonic Ordering

- **Decision:** N ∈ {5,6,7,8} (same set as Costas), direction 50% up / 50% down, Δf uniform in
  [2, 5] MHz (same as Costas), random start frequency constrained to stay inside the Nyquist
  guard band, generic [1, 20] µs pulse width, and phase continuity across chip transitions.
- **Rationale:** Stepped frequency is essentially a **discretised LFM**. Sharing the Costas
  parameter set gives family consistency and makes the two TF signatures directly comparable.
  **The difference from Costas:** Costas frequencies are a random permutation (scattered blocks),
  whereas stepped frequencies are monotonically ordered (a staircase). That distinction gives the
  model a clear visual cue to separate the two. Phase continuity avoids sinc artefacts.
- **Alternatives:** Fixed direction (unrealistic), symmetric baseband (identical to Costas, weakens
  the distinguishing feature), naive phase switching (sinc artefacts).
- **Impact:** TF signature: a monotonic staircase (up or down). Carries a confusion risk with LFM
  (both are monotonic in frequency) — but LFM is continuous while stepped is discrete.

---

## 2026-05-04 — Main Generation Loop: Sequential, Per-Sample Seed, AWGN Kept Separate

- **Decision:** Five design choices for `main_generate_dataset.m`:
  1. **Order:** sequential (5000 LFM, then 5000 NLFM, ...). Shuffling happens in Python at split
     time.
  2. **Per-sample seed:** `rng(master_seed + global_sample_idx, 'twister')` before each sample, so
     any single signal can be regenerated in isolation.
  3. **Progress reporting** every 500 samples.
  4. **AWGN kept separate:** the HDF5 stores **clean signals + the target SNR**. The actual noise
     is added in Python at read time. This allows different noise realisations per epoch and keeps
     the noise strategy flexible.
  5. **HDF5 schema:** minimal data + core metadata attributes.
- **Rationale:** Sequential generation is easy to debug, and the split libraries shuffle anyway.
  Per-sample seeding matters because a reviewer may want to inspect one misclassified sample.
  Keeping AWGN out of storage lets the same clean signal be augmented differently each epoch.
  Full per-sample parameter structs are not stored: with per-sample seeding they can be
  regenerated on demand, which saves disk.
- **Alternatives:** Interleaved generation (complex to implement, hard to debug), a single global
  seed (independent regeneration impossible), AWGN inside the main loop (inflexible), storing all
  parameter structs (gigabytes).
- **Impact:** `main_generate_dataset.m` + `utils/save_dataset_h5.m`. Generation takes ~5–15 minutes.

---

## 2026-05-17 — CWD Library Choice: Custom NumPy (tftb v0.2.0 Has No CWD)

- **Decision:** A **custom NumPy implementation** of the Choi–Williams Distribution was written
  (`preprocessing/time_frequency/cwd.py`). `tftb` is still used for WVD validation, but not for CWD.
- **Rationale:** It had been assumed that `tftb` v0.2.0 contained a `ChoiWilliamsDistribution`
  class. Direct inspection of the package showed it does not — only a docstring reference exists,
  not an implementation. The custom implementation is mathematically small (~200 lines) and was
  validated against `tftb`'s `WignerVilleDistribution` in the σ→∞ limit, achieving a **Pearson
  correlation of 1.0** (bit-for-bit identical total energy and maximum). This is a stronger
  guarantee than a library-to-library comparison, and it is defensible academically: "we
  implemented CWD from first principles following Choi & Williams (1989), verified against tftb's
  WignerVilleDistribution in the σ→∞ limit."
- **Alternatives:** Fork `tftb` and add CWD (maintenance burden), use an old `tftb` 0.1.x
  (Python 3.11 / NumPy 2.x incompatibility risk), another library (`pytftb`, `tfa` — less
  maintained).
- **Impact:** `cwd.py` computes CWD in the time-lag formulation. The manuscript's Methods section
  will note this implementation detail in 2–3 sentences.

---

## 2026-05-17 — CWD Sigma Parameter: σ = 1.0

- **Decision:** σ = 1.0 for the Choi–Williams Gaussian-product kernel (`DEFAULT_SIGMA` in `cwd.py`).
- **Rationale:** σ is the kernel's only tunable parameter: small σ gives strong cross-term
  suppression but blurs auto-terms; large σ leaves cross-terms unsuppressed (approaching the WVD
  limit). Our signals are **mono-component** (one radar class per sample), so the cross-term
  problem is far milder than for multi-component signals. σ = 1.0 is the balanced default from
  Choi & Williams (1989) and matches MATLAB TFTB's `tfrcw`. It answers "why this value?" cleanly.
  Visual validation across all 8 classes was successful: cross-terms below −50 dB, sharp
  auto-terms.
- **Alternatives:** σ = 0.1 (aggressive suppression, unnecessary here), σ = 3.0 or 10 (cross-terms
  under-suppressed, CWD's advantage is lost), a σ sweep (unnecessary experimental cost).
- **Impact:** `cwd.py` takes `sigma` as an optional argument, defaulting to 1.0.

---

## 2026-05-17 — CWD Downsampling: (256, 64), Consistent with STFT

- **Decision:** CWD output size is fixed at (n_freq = 256, n_time = 64), computed with
  `time_step = 32`, `n_freq = 256`, `max_lag = n_freq // 2 - 1 = 127`.
- **Rationale:** A full-resolution CWD (2048×2048) costs 32 MB per sample — pre-computing the
  dataset would need ~1.28 TB, and even on-the-fly it would take 50+ seconds per call, stalling
  training. After downsampling, one sample is 64 KB and takes ~33 ms. This lands in the **same
  resolution class** as the STFT output, so the same CNN/ResNet/ViT accepts both representations
  at an identical input shape. That is critical for an apples-to-apples comparison: the models
  compare the **information content** of the representations, not their resolution. The
  `max_lag = n_freq // 2 - 1` formula guarantees positive and negative lag regions stay disjoint
  in the FFT buffer (Hermitian symmetry).
- **Alternatives:** Full (2048, 2048) (RAM/speed), 224×224 (inconsistent with the STFT grid),
  (512, 128) (a middle ground, but inconsistent with STFT), variable resolution per class (breaks
  the comparison).
- **Impact:** A shared `tf_to_image()` converts STFT/CWD/WVD outputs to a common 224×224 image.

---

## 2026-05-17 — Runtime AWGN Function: NumPy, Active-Region Power, Explicit RNG

- **Decision:** `preprocessing/noise/awgn.py` with a single public function
  `add_awgn(signal, snr_db, active_idx, rng)` — the Python counterpart of the MATLAB
  `add_awgn.m`. Signal power is measured over the active region; the noise is complex
  (real and imaginary each `N(0, noise_power/2)`); an explicit `rng` is mandatory.
- **Rationale:** Follows the "AWGN on the fly" decision — noise is not pre-computed but added at
  each DataLoader call. Using active-region power is critical: the zero padding would otherwise
  depress the measured signal power and skew the achieved SNR (measured deviation of ≈ +3 dB at a
  0 dB target). An explicit `rng` is required because DataLoader workers must each run with their
  own seed; the global `np.random` state is not shared across worker processes, so reproducibility
  would break. The per-component variance convention matches MATLAB exactly.
- **Alternatives:** Global `np.random` state (breaks reproducibility in workers), automatic
  active-region detection inside the function (the caller already knows the indices),
  per-component variance = full noise power (deviates from the MATLAB convention).
- **Impact:** Unit tests over 50 trials × 6 SNR levels show target-vs-empirical SNR deviation
  ≤ 0.13 dB std. Same seed → identical output (max abs diff = 0.00).

---

## 2026-05-17 — PyTorch RadarPulseDataset + DataLoader: On-the-Fly, Per-Sample Seeded, Multi-Worker Safe

- **Decision:** `preprocessing/datasets/radar_pulse_dataset.py` provides `RadarPulseDataset` and
  the `radar_pulse_worker_init` worker init function. Each `__getitem__` performs: HDF5 read →
  AWGN at the target SNR → TF representation (chosen once in `__init__`) → `tf_to_image` →
  `(1, 224, 224)` float32 tensor + integer label. Per-sample seeding uses
  `master_seed + global_sample_idx`. The HDF5 handle is opened lazily (h5py is not fork-safe) and
  reset per worker.
- **Rationale:** Consistent with the on-the-fly AWGN decision. Per-sample seeding makes
  SNR-stratified evaluation reproducible: the same `(idx, master_seed)` always yields the same
  output. Selecting `tf_repr` in `__init__` means each of the nine experiments builds its own
  dataset/loader pair — memory-efficient and clean. The worker init function both reopens the HDF5
  handle per worker process and seeds the per-worker RNGs.
- **Alternatives:** Per-epoch seeding inside the dataset (added later in the trainer instead),
  a single dataset serving all three TF representations at `__getitem__` time (flexible but a
  messy API), eager loading of the whole dataset into RAM (feasible at ~280 MB but riskier under
  cloud memory limits).
- **Impact:** Verified across 12 unit/integration tests. Multi-process output is **bit-for-bit
  identical** to single-process (max abs diff = 0.00).

---

## 2026-05-17 — DataLoader Throughput Benchmark

- **Decision:** `num_workers = 4, batch_size = 64` for all three TF representations.
- **Rationale:** A realistic benchmark (12 configurations: 3 TF × 2 worker counts × 2 batch sizes)
  showed the same configuration is optimal for all three representations — so a single
  hyperparameter setup applies to all trainings, which is ideal for a controlled academic
  comparison. Multi-worker speedup was a consistent ~2.2×, and batch = 64 beat batch = 32 in every
  configuration. This also revised the earlier "cloud is mandatory" assumption: the local RTX 5050
  turned out to be within a workable range.
- **Alternatives:** Larger batches (128, 256) — VRAM-constrained; `num_workers = 8` — marginal
  gain on a laptop core count; a pre-computed pipeline — rejected earlier (24 GB of disk, loses
  augmentation).
- **Impact:** All nine experiments use the same DataLoader configuration.

---

## 2026-05-17 — TF-to-Image Transform: dB Scale + Per-Sample Max-Normalise + 224×224, 1 Channel

- **Decision:** `preprocessing/transforms/tf_to_image.py` exposes
  `tf_to_image(tf_magnitude, output_size=(224, 224), db_floor=-60.0, ...)`. Pipeline: raw
  magnitude → dB (peak-relative) → clip to [db_floor, 0] → remap to [0, 1] → bilinear resize →
  float32 single channel. All three representations therefore yield the same `(1, 224, 224)` tensor.
- **Rationale:** **(a)** Per-sample max-normalisation is robust to AWGN (relative to peak rather
  than absolute power) and keeps train/test consistent. Global z-scoring was rejected because AWGN
  changes each sample's mean/std and would create a distribution shift. **(b)** A single channel:
  TF magnitude is naturally greyscale, and copying it to 3 channels wastes memory (timm adapts the
  stem to 1 channel instead). **(c)** 224×224 is the standard input for ResNet/ViT. **(d)** A
  −60 dB floor matches the value used in all visual validation. **(e)** Bilinear interpolation is a
  safe default for the mixed up/down resampling involved.
- **Alternatives:** Global z-score (distribution shift under AWGN), 3-channel input (multi-
  representation fusion is a separate research question), area interpolation (only appropriate for
  pure downsampling).
- **Impact:** Verified across 7 unit tests plus a visual 3×8 grid (8 classes × 3 representations)
  that is a candidate centrepiece for the manuscript's Methods section. Aggregate per-representation
  statistics differ (STFT mean = 0.12, CWD 0.10, WVD 0.24), i.e. the three representations present
  genuinely different statistical structure to the model.

---

## 2026-05-18 — Split Module (SUPERSEDED)

> ⚠️ **SUPERSEDED by the 2026-05-22 entry below, and removed from the repository during the
> 2026-07-14 cleanup.** This entry is retained for the historical record only.

- **Original decision:** A `preprocessing/splits/` module (`make_splits.py`, `load_splits.py`,
  28 tests) producing `.npy` index files under `data_generation/synthetic_samples/splits/`, with
  joint (class, SNR) stratification and a SHA256 dataset-hash guardrail. It produced a
  27,941 / 6,003 / 6,056 split.
- **Why it was superseded:** The training pipeline that was actually built
  (`experiments/train.py`, 2026-05-22) reads a different artefact — `configs/splits.npz`, produced
  by `scripts/make_splits.py`, giving an exact 28,000 / 6,000 / 6,000 split. The two splits are
  **not the same** (their test sets overlap by only ~15%). The `preprocessing/splits/` module was
  never imported by any training or evaluation script, so it was dead code that additionally
  misrepresented which split the published results were produced on. It was deleted.
- **Retained idea worth revisiting:** the SHA256 dataset-hash guardrail (catching "dataset was
  regenerated but the splits are stale") is a good practice and is **not** currently implemented in
  `scripts/make_splits.py`; the `.npz` does store a `dataset_fingerprint`, but it is not verified at
  load time.

---

## 2026-05-22 — Custom CNN: VGG-Style, 5 Blocks, ~1.77M Parameters (Baseline)

- **Decision:** A compact VGG-style CNN as the baseline architecture (`models/custom_cnn.py`):
  - **Input:** `(B, 1, 224, 224)` — the `tf_to_image` output, identical for all three representations
  - **5 conv blocks:** blocks 1–4 are double conv (Conv-BN-ReLU ×2), block 5 is single conv.
    Channels 1→32→64→128→256→256.
  - **MaxPool(2×2) after each block:** spatial flow 224→112→56→28→14→7
  - **Head:** global average pooling → Dropout(0.5) → Linear(256, 8)
  - **Output:** 8 logits (no softmax; `nn.CrossEntropyLoss` applies log-softmax internally)
  - **Parameters:** 1,765,032 (~1.77M). Almost all of them are in the feature extractor; the
    classifier head holds only 2,056.
- **Rationale:** The custom CNN's academic role is a **fair baseline** — compact enough to show
  how much ResNet-50 and ViT actually buy at low SNR, but strong enough to reach reasonable
  accuracy. ~1.8M parameters is consistent with radar TF baselines in the literature. BatchNorm
  keeps it consistent with ResNet (fair comparison). GAP + FC (instead of flatten + a large FC)
  sharply reduces overfitting risk. ReLU is the natural baseline choice (GELU would imitate the
  ViT and compromise the baseline's role).
- **Alternatives:** A minimal 4-block ~0.5M net (too weak, creates a "trivial vs strong" story),
  a 6-block bottleneck ~5–8M net (approaches ResNet, weakening the baseline role),
  GroupNorm/LayerNorm (overkill for a baseline), flatten + FC (parameter-heavy, overfits).
- **Impact:** `models/custom_cnn.py` + 12 unit tests (all passing), covering forward shape,
  gradient flow, determinism, and seed reproducibility. The same architecture accepts all three
  representations at an identical input shape.

---

## 2026-05-22 — Training Infrastructure: Config-Driven Trainer, Frozen Split, Per-Epoch Reseeding

- **Decision:** All experiments run through a single architecture-agnostic trainer:
  - **Frozen split:** `scripts/make_splits.py` → `configs/splits.npz` (28,000 / 6,000 / 6,000,
    joint (label, SNR) stratification, seed = 42). Every experiment and all downstream analysis
    reads this same split.
  - **Config system:** `experiments/config.py` (dataclass + YAML); one YAML per experiment
    (`configs/<tf>_<arch>.yaml`).
  - **Model registry:** `models/registry.py` maps `cfg.model.name` → class. Adding an architecture
    is a one-line change.
  - **Trainer:** `experiments/train.py`. AdamW + linear warmup (3 epochs) + cosine decay,
    cross-entropy, 50 epochs, early stopping on validation loss (patience 10), AMP on CUDA.
  - **Per-epoch reseeding (train only):** the training dataset is rebuilt each epoch with
    `master_seed = base + (epoch+1) * 1_000_003`, giving a different AWGN realisation per epoch —
    genuine on-the-fly augmentation. Validation and test use a fixed seed, so they stay
    deterministic and reproducible.
  - **Outputs:** `experiments/checkpoints/<name>/{best,last}.pth` and
    `experiments/results/<name>/{config.yaml, history.json, tb/}`.
- **Rationale:** One config-driven trainer means all nine experiments share the same code — an
  apples-to-apples comparison. The frozen split guarantees every experiment sees the same test set.
  Per-epoch reseeding is what actually delivers the intended "avoid overfitting to a fixed noise
  pattern" property: the dataset alone would otherwise hold the training noise fixed. Fixed val/test
  seeds are required for bit-for-bit reproducible SNR-stratified evaluation.
- **Alternatives:** A separate script per experiment (code duplication); computing the split at
  runtime (drift risk, no guarantee all experiments share a split); fixed training noise (loses the
  augmentation benefit); embedding the epoch seed inside the dataset (cleaner to handle in the
  trainer).
- **Impact:** Validated end-to-end on a mock dataset: training loop, LR schedule, per-epoch reseed
  (train varies, val deterministic), checkpointing, early stopping, TensorBoard logging.

---

## 2026-05-23 — ResNet-50: ImageNet-Pretrained, 1-Channel Adaptation, lr = 1e-4

- **Decision:** The second architecture is ResNet-50 (timm), ImageNet-1k pretrained, with
  `in_chans=1` (timm collapses the 3-channel stem weights to 1 channel, the standard
  information-preserving method) and `num_classes=8`. Fine-tuning uses **lr = 1e-4** instead of
  3e-4; everything else matches the custom CNN. ~23.5M parameters.
- **Rationale:** Transfer learning is standard in the radar TF classification literature, and it
  reflects the realistic practitioner question: "small custom net from scratch vs a strong
  pretrained backbone." lr = 1e-4 adapts the pretrained weights without destroying them.
- **Alternatives:** ResNet from scratch (fully fair against the custom CNN, but risks overfitting
  with 28k samples), replicating 1 channel to 3 (3× memory for no benefit), lr = 3e-4 (aggressive
  for a pretrained model).
- **Impact:** `models/resnet50.py` + one registry line + configs.

---

## 2026-05-23 — ViT-Small: ImageNet-Pretrained, Capacity-Matched, lr = 5e-5

- **Decision:** The third architecture is ViT-Small (timm `vit_small_patch16_224`), ImageNet-1k
  pretrained, ~21.5M parameters — deliberately chosen to be close to ResNet-50's ~23.5M.
  `in_chans=1`, `num_classes=8`, lr = 5e-5 (lower than the CNNs), gradient clipping at 1.0.
- **Rationale:** Matching ViT-Small's capacity to ResNet-50 isolates the comparison along the
  **paradigm** axis (self-attention vs convolution) rather than raw model size — using ViT-Base
  (86M) would confound capacity with paradigm and would be hard to fine-tune reliably on 28k
  samples. ViTs lack convolutional inductive bias and are more sensitive to optimisation on
  small/medium data, hence the lower learning rate and gradient clipping.
- **Alternatives:** ViT-Base (confounds capacity with paradigm), Swin-Tiny (hierarchical, blurs the
  "pure ViT" story), ViT from scratch (very hard with 28k samples), lr = 1e-4 (aggressive for ViT).
- **Impact:** `models/vit.py` + one registry line + configs. The expectation was explicit: if the
  ViT does not beat the CNNs, that strengthens the "a compact CNN is optimal for this task"
  message; if it does, that is a finding in itself. (The former turned out to be the case.)

---

## 2026-05-25 — CWD Training Time: ~740 s/epoch (Local RTX 5050)

- **Decision:** CWD trainings run locally. Measured ~740 s/epoch for the custom CNN; a full
  50-epoch run is ~10 hours, so three CWD trainings are ~30 hours.
- **Rationale:** Thanks to the decimated CWD implementation, CWD is only ~2.7× slower than STFT
  (740 s vs 270 s) rather than the ~16× a full-resolution implementation would cost. The effort of
  moving to Kaggle (dataset upload, reproducibility guarantees) is not worth it at this duration.
- **Impact:** The three CWD configs differ only in `tf_repr`. Runs can be chained in a single block.

---

## 2026-07-13 — WVD Evaluation Gap and the Analysis Suite

- **Decision:** The three WVD models had been trained (50 epochs each, checkpoints saved) but never
  evaluated: `experiments/run_wvd_all.sh` only invoked `train.py`, never `evaluate.py`, so their
  `test_metrics.json` and figures were missing. The evaluation was run, the runner script was fixed
  to chain `evaluate.py` after each training, and a full analysis suite was added under `analysis/`.
- **Rationale:** Without the WVD test metrics the 3×3 experiment matrix was incomplete and the
  central research question (does WVD's cross-term behaviour hurt at low SNR?) was unanswered.
- **Impact:** Four analysis scripts, each regenerable from committed artefacts:
  - `compare_all_experiments.py` — 9-model SNR robustness figures + summary table
  - `wvd_lowsnr_confusion.py` — WVD's per-SNR confusion breakdown
  - `model_complexity.py` — parameter and FLOP counts
  - `statistical_significance.py` — Wilson CIs, bootstrap, two-proportion z-test, paired McNemar
  - `qualitative_tf_illustration.py` — STFT/CWD/WVD × SNR grid (visual proof of the cross-term
    mechanism)

  Headline results (see `docs/results_summary.md`): STFT (97.9%) > CWD (97.4%) ≫ WVD (89.0%), with
  the entire gap concentrated at low SNR (60 points at −10 dB, ~0 above +2 dB).

---

## 2026-07-14 — Repository Cleanup: English-Only Documentation, Dead Code Removed

- **Decision:** The repository was brought to a clean, publishable state:
  - All documentation rewritten in English (`README.md`, `docs/decisions.md`,
    `docs/results_summary.md`, new `docs/dataset.md`).
  - `docs/project_context.md` deleted — it was a chat-priming/onboarding file rather than project
    documentation, was badly out of date (it described the experiment matrix as pending), and was
    factually wrong about which split the pipeline uses. Its genuinely useful technical content
    (HDF5 storage convention, class table, reading patterns) was migrated to `docs/dataset.md`.
  - `preprocessing/splits/` and the `.npy` split files deleted (see the 2026-05-18 entry).
  - Superseded single-purpose analysis figures removed, now that the 9-experiment figures cover them.
  - Internal build-log jargon ("Module A/B/C", "Phase 2b") removed from code docstrings so the
    repository reads as a standalone research project rather than a development log.
- **Rationale:** The project is heading to publication and a public repository. Dead code that
  contradicts the published results (the divergent split) is a reproducibility hazard, and
  Turkish-language internal planning documents are not useful to an external reader.
- **Impact:** Documentation is English-only and internally consistent. `configs/splits.npz` (via
  `scripts/make_splits.py`) is now the single, unambiguous source of the frozen split.

---

## Appendix — Resolved Open Questions

All questions raised during the design phase have been closed:

| Question | Resolution | Date |
|---|---|---|
| Sample rate: 100 or 200 MHz? | 100 MHz | 2026-05-04 |
| Pulse width range | 1–20 µs, fixed 2048-sample frame | 2026-05-04 |
| Samples per class | 5000, SNR assigned per sample | 2026-05-04 |
| SNR step size | 2 dB, 16 points | 2026-05-04 |
| Train/val/test split | 70/15/15 | 2026-05-04 |
| Class balance | Perfectly balanced | 2026-05-04 |
| File format | HDF5 (.h5) | 2026-05-04 |
| Random seed management | Global 42, layered | 2026-05-04 |
| Padding strategy | Random position | 2026-05-04 |
| P1–P4 handling | One combined class, 25% each | 2026-05-04 |
| Barker code lengths | B7 + B11 + B13, rectangular chip | 2026-05-04 |
| Costas sequence length | N ∈ {5,6,7,8}, 2 sequences per N | 2026-05-04 |
| Carrier frequency range | Complex baseband, 5% guard band | 2026-05-04 |
| Where AWGN is added | Full frame after padding, SNR from active-region power | 2026-05-04 |
| CWD parameters | Custom NumPy impl, σ = 1.0, (256, 64) | 2026-05-17 |
| WVD parameters | σ → ∞ via the CWD implementation | 2026-05-17 |
| Runtime AWGN function | `preprocessing/noise/awgn.py` | 2026-05-17 |
| dB scaling + normalisation | Per-sample max-normalise, −60 dB floor | 2026-05-17 |
| Image size / channels | 224×224, 1 channel | 2026-05-17 |
| Dataset/DataLoader design | `RadarPulseDataset` + per-sample seeding | 2026-05-17 |
