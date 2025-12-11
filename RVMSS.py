#!/usr/bin/env python3

import statistics
import math

from model import run_model       # stochastic model
from benchmark import run_deterministic_benchmark   # deterministic benchmark


# ---------------------------------------------------------------------------
# FUNCTION: run_vrmss_experiment
# ---------------------------------------------------------------------------

def run_vrmss_experiment(
    time_str: str,
    n: int,
    num_successful_runs: int = 5,
    base_seed: int | None = None,
    verbose_runs: bool = False,
    **run_model_kwargs,
):
    """
    Runs the stochastic model and deterministic benchmark multiple times
    using identical seeds per run. Continues until EXACTLY `num_successful_runs`
    deterministic runs succeed. Skips seeds where the deterministic model fails.

    Returns:
        {
            "num_requested_successful_runs": int,
            "num_successful_runs": int,
            "attempts_made": int,
            "n_scenarios": int,
            "seeds_successful": list[int | None],
            "seeds_skipped": list[int | None],
            "Z_sto_all": list[float],
            "Z_det_all": list[float],
            "VRMSS_all": list[float],
            "avg_Z_sto": float,
            "avg_Z_det": float,
            "avg_VRMSS": float,
        }
    """

    Z_sto_list = []
    Z_det_list = []
    VRMSS_list = []
    seeds_successful = []
    seeds_skipped = []

    print("\n=== START VRMSS EXPERIMENT ===")
    print(f"time_str = {time_str}")
    print(f"Requested successful runs = {num_successful_runs}")
    print(f"Base seed = {base_seed}\n")

    successful_count = 0
    attempts = 0
    max_attempts = 1000  # safety limit

    # ------------------------------------------------------------------
    # Loop until required number of successful runs is achieved
    # ------------------------------------------------------------------
    while successful_count < num_successful_runs:

        # Safety stop to prevent infinite bad-seed loops
        if attempts > max_attempts:
            raise RuntimeError(
                f"Exceeded {max_attempts} attempts without reaching "
                f"{num_successful_runs} successful deterministic runs."
            )

        # Determine seed
        seed = None
        if base_seed is not None:
            seed = base_seed + attempts

        # Track total attempts
        attempts += 1

        # --- 1) Deterministic evaluation ---
        try:
            Z_det_scalar = run_deterministic_benchmark(
                time_str=time_str,
                n=n,
                seed=seed,
            )
        except Exception as e:
            print(f"[ATTEMPT {attempts}] seed={seed} -> deterministic FAILED ({e}); skipping.\n")
            seeds_skipped.append(seed)
            continue

        # Check deterministic output validity
        if Z_det_scalar is None or Z_det_scalar == 0 or math.isnan(Z_det_scalar) or math.isinf(Z_det_scalar):
            print(f"[ATTEMPT {attempts}] seed={seed} -> deterministic invalid ({Z_det_scalar}); skipping.\n")
            seeds_skipped.append(seed)
            continue

        # --- 2) Stochastic model ---
        res_sto = run_model(
            time_str=time_str,
            n=n,
            seed=seed,
            verbose=verbose_runs,
            **run_model_kwargs,
        )

        model_sto = res_sto["model"]
        Z_sto = model_sto.objVal

        # --- 3) Store results ---
        Z_sto_list.append(Z_sto)
        Z_det_list.append(Z_det_scalar)
        seeds_successful.append(seed)

        # --- 4) VRMSS ---
        vrmss = (Z_sto - Z_det_scalar) / Z_det_scalar
        VRMSS_list.append(vrmss)

        successful_count += 1

        print(f"[SUCCESS {successful_count}/{num_successful_runs}] seed={seed}")
        print(f"  Z_sto  = {Z_sto:.6f}")
        print(f"  Z_det  = {Z_det_scalar:.6f}")
        print(f"  VRMSS  = {vrmss:.6f}\n")

    # ------------------------------------------------------------------
    # Final Stats
    # ------------------------------------------------------------------
    avg_Z_sto = statistics.fmean(Z_sto_list)
    avg_Z_det = statistics.fmean(Z_det_list)
    avg_VRMSS = statistics.fmean(VRMSS_list)

    print("=== VRMSS SUMMARY ===")
    print(f"Requested successful runs : {num_successful_runs}")
    print(f"Successful runs          : {successful_count}")
    print(f"Total attempts made      : {attempts}")
    print(f"Seeds (successful)       : {seeds_successful}")
    if seeds_skipped:
        print(f"Seeds (skipped)          : {seeds_skipped}")
    print(f"Avg Z^sto                : {avg_Z_sto:.6f}")
    print(f"Avg Z^det                : {avg_Z_det:.6f}")
    print(f"Avg VRMSS                : {avg_VRMSS:.6f}")
    print("======================\n")

    return {
        "num_requested_successful_runs": num_successful_runs,
        "num_successful_runs": successful_count,
        "attempts_made": attempts,
        "n_scenarios": n,
        "seeds_successful": seeds_successful,
        "seeds_skipped": seeds_skipped,
        "Z_sto_all": Z_sto_list,
        "Z_det_all": Z_det_list,
        "VRMSS_all": VRMSS_list,
        "avg_Z_sto": avg_Z_sto,
        "avg_Z_det": avg_Z_det,
        "avg_VRMSS": avg_VRMSS,
    }

if __name__ == "__main__":
    time_str = "2025-10-07 12:00:00+00:00"
    n = 6
    num_successful_runs = 5
    base_seed = 15

    results = run_vrmss_experiment(
        time_str=time_str,
        n=n,
        num_successful_runs=num_successful_runs,
        base_seed=base_seed,
    )
