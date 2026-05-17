"""
PyTorch ``Dataset`` for the radar-pulse dataset.

This is the runtime pipeline that connects everything Module B built:

    HDF5 read  ->  AWGN at sample's intended SNR (runtime)
              ->  TF representation (STFT / CWD / WVD, on-the-fly)
              ->  tf_to_image (dB-clip + normalize + resize 224x224)
              ->  PyTorch float32 tensor (1, 224, 224) + integer label

Design choices (decisions.md, 2026-05-17 Dataset entry)
-------------------------------------------------------
- **Lazy HDF5 access.** ``h5py.File`` is opened per-worker on first
  ``__getitem__`` call, not in ``__init__`` (h5py file handles are NOT
  fork-safe; opening in ``__init__`` would break with
  ``num_workers > 0`` on Linux). PyTorch's recommended pattern.
- **Per-sample seeding.** Each sample's noise realisation is keyed on
  ``master_seed + global_sample_idx``, so the (idx, seed) pair fully
  determines what the model sees. This makes Module D's SNR-stratified
  evaluation bit-for-bit reproducible: same indices in eval mode ->
  same noise -> same predictions across runs.
- **No epoch-level augmentation by default.** Adding ``epoch_seed`` is
  trivial later — caller passes a different ``master_seed`` per epoch
  (or we add an ``epoch`` arg) — but the default behaviour is fixed
  per-sample for evaluation reproducibility.
- **TF representation chosen at construction time.** A single Dataset
  instance commits to one of {stft, cwd, wvd}. Module C will create
  three separate Dataset/Loader pairs, one per representation, which
  matches the 3-mimari x 3-gosterim = 9 experiment matrix cleanly.

Public API
----------
- :class:`RadarPulseDataset` — the Dataset class.
- :func:`radar_pulse_worker_init` — DataLoader ``worker_init_fn``.

Author: Kaan Emre Evci
Project: Radar Pulse Classification (Module B, Phase 2b)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.noise.awgn import add_awgn
from preprocessing.time_frequency.cwd import compute_cwd
from preprocessing.time_frequency.stft import compute_stft
from preprocessing.time_frequency.wvd import compute_wvd
from preprocessing.transforms.tf_to_image import tf_to_image

__all__ = ["RadarPulseDataset", "radar_pulse_worker_init"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
VALID_TF_REPRS = ("stft", "cwd", "wvd")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class RadarPulseDataset(Dataset):
    """
    PyTorch Dataset wrapping the synthetic HDF5 radar pulse dataset.

    Parameters
    ----------
    h5_path : str or Path
        Path to ``dataset.h5`` (the file produced by Module A).
    indices : array_like of int
        Global sample indices (within ``[0, N_total)``) that this Dataset
        instance covers. Caller is responsible for the train/val/test
        split — typically via ``sklearn.model_selection.train_test_split``
        with ``stratify=labels``. See decisions.md "Train/Val/Test
        Bolunmesi: 70/15/15".
    tf_repr : {"stft", "cwd", "wvd"}
        Which time-frequency representation to compute.
    add_noise : bool, default True
        If True, AWGN is added at each sample's ``snr_db`` (read from
        HDF5). Set False to feed clean signals (debugging, sanity checks).
    master_seed : int, default 42
        Master RNG seed. The actual per-sample noise seed is
        ``master_seed + global_sample_idx`` (NOT the position in
        ``indices``, but the original HDF5 index — so the same sample
        gets the same noise regardless of train/val split position).
    output_size : (H, W), default (224, 224)
        Image dimensions handed to the model.
    db_floor : float, default -60.0
        Lower dB clip for ``tf_to_image``.

    Notes
    -----
    - Use with ``DataLoader(num_workers=N, worker_init_fn=radar_pulse_worker_init)``.
      The init function ensures each worker process has its own h5py
      file handle (h5py is NOT fork-safe).
    - ``__len__`` returns ``len(indices)``, NOT the full dataset size.

    Examples
    --------
    >>> indices = np.arange(40000)
    >>> ds = RadarPulseDataset(
    ...     h5_path="data_generation/synthetic_samples/dataset.h5",
    ...     indices=indices,
    ...     tf_repr="cwd",
    ...     add_noise=True,
    ...     master_seed=42,
    ... )
    >>> img, label = ds[0]
    >>> img.shape, img.dtype, type(label)
    (torch.Size([1, 224, 224]), torch.float32, <class 'int'>)
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        indices: np.ndarray,
        tf_repr: str,
        add_noise: bool = True,
        master_seed: int = 42,
        output_size: Tuple[int, int] = (224, 224),
        db_floor: float = -60.0,
    ) -> None:
        if tf_repr not in VALID_TF_REPRS:
            raise ValueError(
                f"tf_repr must be one of {VALID_TF_REPRS}, got {tf_repr!r}."
            )

        self.h5_path = str(Path(h5_path).resolve())
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError(f"indices must be 1-D, got shape {self.indices.shape!r}.")
        if self.indices.size == 0:
            raise ValueError("indices is empty.")

        self.tf_repr = tf_repr
        self.add_noise = bool(add_noise)
        self.master_seed = int(master_seed)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.db_floor = float(db_floor)

        # h5py file handle: opened LAZILY in __getitem__ on first call.
        # Critical for multi-worker DataLoader (h5py is not fork-safe).
        self._h5_file: Optional[h5py.File] = None

        # Cache static metadata read once in __init__ (cheap).
        # These do not require an open file handle in __getitem__.
        with h5py.File(self.h5_path, "r") as f:
            self._fs = float(np.asarray(f.attrs["sample_rate_hz"]).ravel()[0])
            self._signal_length = int(np.asarray(f.attrs["signal_length"]).ravel()[0])

    # -- File handle management ---------------------------------------------
    def _ensure_open(self) -> h5py.File:
        """Open the HDF5 file on first call; reuse the handle afterwards."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __del__(self) -> None:
        """Close the file handle when this Dataset instance is garbage-collected."""
        try:
            if self._h5_file is not None:
                self._h5_file.close()
        except Exception:
            # __del__ must never raise.
            pass

    # -- Dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, local_idx: int) -> Tuple[torch.Tensor, int]:
        """
        Return ``(image, label)`` for sample at position ``local_idx`` in
        ``self.indices``.

        Returns
        -------
        image : torch.Tensor
            Shape (1, H, W), dtype float32, range [0, 1].
        label : int
            Class index in [0, 8).
        """
        global_idx = int(self.indices[local_idx])
        f = self._ensure_open()

        # -- 1. Read clean signal (MATLAB column-major layout) ---------
        ri = np.asarray(f["signals"][:, :, global_idx]).T  # (N, 2) float32
        signal = (ri[:, 0] + 1j * ri[:, 1]).astype(np.complex64)

        # -- 2. Read label and intended SNR ----------------------------
        label = int(np.asarray(f["labels"][:]).ravel()[global_idx])
        snr_db = float(np.asarray(f["snr_db"][:]).ravel()[global_idx])

        # -- 3. Add AWGN at the sample's intended SNR (optional) -------
        if self.add_noise:
            # Detect active region from the clean signal (Module A did
            # not export start/stop indices in the per-sample group;
            # we detect via magnitude threshold for now).
            active_idx = self._detect_active_region(signal)
            # Per-sample seed for full reproducibility (see class docstring).
            sample_seed = self.master_seed + global_idx
            rng = np.random.default_rng(sample_seed)
            signal = add_awgn(
                signal,
                snr_db=snr_db,
                active_idx=active_idx,
                rng=rng,
            )

        # -- 4. Time-frequency representation --------------------------
        if self.tf_repr == "stft":
            # Project compute_stft returns (f, t, Zxx) — freq first.
            _, _, tf_complex = compute_stft(signal, fs=self._fs)
            tf_mag = np.abs(tf_complex)
        elif self.tf_repr == "cwd":
            tf_mag, _, _ = compute_cwd(signal, fs=self._fs)
        elif self.tf_repr == "wvd":
            tf_mag, _, _ = compute_wvd(signal, fs=self._fs)
        else:
            # already validated in __init__, but defensive
            raise RuntimeError(f"unknown tf_repr {self.tf_repr!r}")

        # -- 5. Convert to model input image ---------------------------
        img = tf_to_image(
            tf_mag,
            output_size=self.output_size,
            db_floor=self.db_floor,
        )  # (H, W) float32 in [0, 1]

        # -- 6. To PyTorch tensor with channel dim ---------------------
        tensor = torch.from_numpy(img).unsqueeze(0).contiguous()  # (1, H, W)

        return tensor, label

    # -- Helpers ------------------------------------------------------------
    @staticmethod
    def _detect_active_region(
        signal: np.ndarray,
        threshold: float = 1e-6,
    ) -> Tuple[int, int]:
        """Detect (start_idx, stop_idx) inclusive of the non-zero region.

        Matches MATLAB convention used by ``add_awgn.m``: ``stop_idx``
        is inclusive. If the entire signal is non-zero (e.g. no padding),
        returns ``(0, len(signal) - 1)``.
        """
        mag = np.abs(signal)
        active = np.where(mag > threshold)[0]
        if active.size == 0:
            # Degenerate case; treat the whole frame as "active" so
            # add_awgn does not crash on zero-power input. add_awgn will
            # then raise ValueError, which is the correct behaviour.
            return 0, int(mag.size) - 1
        return int(active[0]), int(active[-1])


# ---------------------------------------------------------------------------
# DataLoader worker_init_fn
# ---------------------------------------------------------------------------
def radar_pulse_worker_init(worker_id: int) -> None:
    """
    DataLoader ``worker_init_fn`` for :class:`RadarPulseDataset`.

    Two responsibilities:

    1. Force each worker to re-open its own h5py file handle (h5py file
       handles are NOT fork-safe). We do this by setting the cached
       ``_h5_file`` attribute back to None on every Dataset instance
       reachable from this worker.
    2. Seed NumPy/Python/torch RNGs deterministically per worker. The
       primary RNG used inside ``add_awgn`` is *not* affected (it is
       seeded per-sample), but other libraries called by future
       augmentation layers may benefit.

    The DataLoader passes this function the worker_id.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        # Not running in a worker (num_workers=0); nothing to do.
        return

    # Reset cached h5py handles on the Dataset instance(s) this worker holds.
    ds = worker_info.dataset
    if isinstance(ds, RadarPulseDataset):
        ds._h5_file = None
    elif hasattr(ds, "datasets"):
        # Handle torch.utils.data.ConcatDataset and similar wrappers.
        for sub in getattr(ds, "datasets", []):
            if isinstance(sub, RadarPulseDataset):
                sub._h5_file = None

    # Seed library-level RNGs per-worker (defensive — not relied upon by
    # add_awgn, which has its own per-sample seeded Generator).
    base_seed = worker_info.seed % (2**31)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
