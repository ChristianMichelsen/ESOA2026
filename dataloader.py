# %%

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class BoIteration:
    sites: np.ndarray
    values_norm: np.ndarray

    grid_sites: np.ndarray
    grid_norm: np.ndarray

    acqf_grid_sites: np.ndarray
    acqf_grid_predictions: np.ndarray


@dataclass
class Data:
    bo: list[BoIteration]
    _bo_surrogate_norm_range_cache: tuple[float, float] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def get_bo_surrogate_norm_range(self) -> tuple[float, float]:
        if self._bo_surrogate_norm_range_cache is not None:
            return self._bo_surrogate_norm_range_cache

        assert isinstance(self.bo[0], BoIteration)
        surrogate_values = np.concatenate([bo.grid_norm.ravel() for bo in self.bo])

        if surrogate_values.size == 0:
            msg = "No surrogate norm values found to compute range."
            raise ValueError(msg)

        if not np.isfinite(surrogate_values).all():
            surrogate_values = surrogate_values[np.isfinite(surrogate_values)]

        self._bo_surrogate_norm_range_cache = (
            float(surrogate_values.min()),
            float(surrogate_values.max()),
        )
        return self._bo_surrogate_norm_range_cache


def load_key(f: Any, key: str) -> np.ndarray:
    return np.asarray(f[key][:])


def load_data(path: Path) -> Data:
    with h5py.File(path, "r") as f:
        i = 1
        bo_iterations: list[BoIteration] = []
        while True:
            try:
                prefix = f"2_bo_iterations/iter_{i:06d}"

                s_bo_samples = f"{prefix}/2_model_update/objective/samples"
                bo_samples_sites = load_key(f, f"{s_bo_samples}/sites")
                bo_samples_values_norm = load_key(f, f"{s_bo_samples}/values_norm")

                s_bo_obj = f"{prefix}/1_acq_optim/acq_obj_pred"
                bo_obj_sites = load_key(f, f"{s_bo_obj}/grid_sites")
                bo_obj_norm = load_key(f, f"{s_bo_obj}/predicted_norm")

                s_bo_acqf = f"{prefix}/1_acq_optim/acq_function"
                bo_acqf_grid_sites = load_key(f, f"{s_bo_acqf}/grid_sites")
                bo_acqf_grid_values = load_key(f, f"{s_bo_acqf}/grid_values")

                bo_iteration = BoIteration(
                    sites=bo_samples_sites,
                    values_norm=bo_samples_values_norm,
                    grid_sites=bo_obj_sites,
                    grid_norm=bo_obj_norm,
                    acqf_grid_sites=bo_acqf_grid_sites,
                    acqf_grid_predictions=bo_acqf_grid_values,
                )

                bo_iterations.append(bo_iteration)
                i += 1
            except KeyError:
                break

    data = Data(bo=bo_iterations)
    return data


def encode_npz_payload(arrays: dict[str, Any]) -> str:
    """Encode arrays as a compact ASCII payload for copy/paste transfer."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return base64.b85encode(buf.getvalue()).decode("ascii")


def decode_npz_payload(payload: str) -> dict[str, np.ndarray]:
    """Decode an ASCII payload produced by ``encode_npz_payload``."""
    raw = base64.b85decode(payload.encode("ascii"))
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# %%

if False:
    data = load_data(Path("meta_model_log_himmelblau.h5"))
    data

    data.bo[0].sites.shape  # (2, 20)
    data.bo[0].values_norm.shape  # (20,)
    data.bo[0].grid_sites.shape  # (2, 10201)
    data.bo[0].grid_norm.shape  # (10201,)
    data.bo[0].acqf_grid_sites.shape  # (10201,)
    data.bo[0].acqf_grid_predictions.shape  # (10201,)

    data.bo[0].sites.shape
    data.bo[-1].sites.shape

    N_bad_initial_iterations = 2

    out = {}
    out["sites"] = data.bo[-1].sites
    out["values_norm"] = data.bo[-1].values_norm

    # the first two iterations are bad, so we skip them
    out["grid_sites"] = np.stack(
        [
            bo.grid_sites.reshape(2, 101, 101)
            for bo in data.bo[N_bad_initial_iterations:]
        ],
        axis=0,
    )
    out["grid_norm"] = np.stack(
        [bo.grid_norm.reshape(101, 101) for bo in data.bo[N_bad_initial_iterations:]],
        axis=0,
    )

    out["acqf_grid_sites"] = np.stack(
        [
            bo.acqf_grid_sites.reshape(2, 101, 101)
            for bo in data.bo[N_bad_initial_iterations:]
        ],
        axis=0,
    )
    out["acqf_grid_predictions"] = np.stack(
        [
            bo.acqf_grid_predictions.reshape(101, 101)
            for bo in data.bo[N_bad_initial_iterations:]
        ],
        axis=0,
    )

    # np.savez_compressed("himmelblau.npz", **out)
    payload = encode_npz_payload(out)
    print(payload)

    # Optional round-trip check.
    restored = decode_npz_payload(payload)
    # np.savez_compressed("himmelblau.npz", **restored)


# %%


if False:
    data = load_data(Path("meta_model_log_himmelblau_ei.h5"))

    out = {}
    out["sites"] = data.bo[-1].sites
    out["values_norm"] = data.bo[-1].values_norm

    # the first four iterations are bad, so we skip them
    N_bad_initial_iterations = 5

    out["grid_sites"] = np.stack(
        [
            bo.grid_sites.reshape(2, 101, 101)
            for bo in data.bo[N_bad_initial_iterations:]
        ],
        axis=0,
    )
    out["grid_norm"] = np.stack(
        [bo.grid_norm.reshape(101, 101) for bo in data.bo[N_bad_initial_iterations:]],
        axis=0,
    )

    out["acqf_grid_sites"] = np.stack(
        [
            bo.acqf_grid_sites.reshape(2, 101, 101)
            for bo in data.bo[N_bad_initial_iterations:]
        ],
        axis=0,
    )
    out["acqf_grid_predictions"] = np.stack(
        [
            bo.acqf_grid_predictions.reshape(101, 101)
            for bo in data.bo[N_bad_initial_iterations:]
        ],
        axis=0,
    )

    # np.savez_compressed("himmelblau_ei.npz", **out)
    payload = encode_npz_payload(out)
    print(payload)

    # Optional round-trip check.
    restored = decode_npz_payload(payload)
    np.savez_compressed("himmelblau_ei.npz", **restored)
