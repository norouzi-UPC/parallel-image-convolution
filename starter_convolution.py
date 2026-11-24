"""
starter_convolution.py — Student Starter Template (Local-Image Edition)
-----------------------------------------------------------------------
Parallel 2D Image Convolution & Synchronization
Group submission approved by the professor, as confirmed by MohammadJavad.
Effort: ~1.5h in-class + up to 1.5h self-paced at home.

Required Local Files in Working Directory:
  - `utilities.py`
  - `small_image.png` (600 x 800 grayscale PNG for Milestones 1-3A & Bonus)
  - `large_image.png` (1600 x 2400 grayscale PNG for Milestone 3B scaling)

Instructions:
  - Complete each TODO section in order (Milestones 1 -> 3, plus Optional Bonus).
  - Run this file frequently: built-in `assert` checks verify correctness at
    every milestone before moving to the next one.
  - Keep the `if __name__ == "__main__":` guard intact so `multiprocessing`
    works safely on Windows, macOS, and Linux.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from multiprocessing import Lock, Process, shared_memory
import numpy as np

from utilities import (
    DEFAULT_LARGE_IMAGE,
    DEFAULT_SMALL_IMAGE,
    get_kernels,
    load_grayscale_image,
    plot_comparison,
)

try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ==============================================================================
# MILESTONE 1 (25 pts): Sequential Baseline & Row-Wise Domain Decomposition
# ==============================================================================
def convolve_sequential(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute valid 5x5 2D convolution on interior pixels [2..H-2, 2..W-2)."""
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float64)

    for r in range(2, H - 2):
        for c in range(2, W - 2):
            acc = 0.0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    acc += img[r + dr, c + dc] * kernel[dr + 2, dc + 2]
            out[r, c] = acc

    return out


def make_row_chunks(
    num_workers: int, start_row: int = 2, end_row: int | None = None
) -> list[tuple[int, int]]:
    """Split row range [start_row, end_row) into `num_workers` disjoint slices.

    For 5x5 convolution on an image of height H, call with:
        make_row_chunks(num_workers, start_row=2, end_row=H - 2)
    For full-image operations (such as the Optional Bonus histogram), call with:
        make_row_chunks(num_workers, start_row=0, end_row=H)

    Note: The final chunk must absorb any remainder rows if not evenly divisible.
    """
    if end_row is None:
        raise ValueError("Please pass end_row explicitly.")
    # TODO (Milestone 1): Replace this placeholder with your chunking logic
    raise NotImplementedError("Implement make_row_chunks")


def convolve_row_range(
    img: np.ndarray, kernel: np.ndarray, out: np.ndarray, r_start: int, r_end: int
) -> None:
    """Compute 5x5 convolution ONLY for rows `r` in [r_start, r_end)."""
    _, W = img.shape
    for r in range(r_start, r_end):
        for c in range(2, W - 2):
            acc = 0.0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    acc += img[r + dr, c + dc] * kernel[dr + 2, dc + 2]
            out[r, c] = acc


# ==============================================================================
# MILESTONE 2 (20 pts): Threading Experiment (Testing the GIL)
# ==============================================================================
def convolve_threads(
    img: np.ndarray, kernel: np.ndarray, num_workers: int = 4
) -> np.ndarray:
    """Run row-chunked 5x5 convolution across `num_workers` Python threads."""
    out = np.zeros_like(img)
    # TODO (Milestone 2):
    #   1. Split interior rows [2, H - 2) using `make_row_chunks`
    #   2. Spawn `num_workers` `threading.Thread` objects targeting `convolve_row_range`
    #   3. Start all threads, join all threads, and return `out`
    raise NotImplementedError("Implement convolve_threads")


# ==============================================================================
# MILESTONE 3A (30 pts): Multiprocessing + Zero-Copy SharedMemory
# ==============================================================================
def _mp_shm_worker(
    shm_img_name: str,
    shm_out_name: str,
    shape: tuple[int, int],
    kernel: np.ndarray,
    r_start: int,
    r_end: int,
) -> None:
    """Worker process: attaches to existing SharedMemory buffers and computes slice."""
    # TODO (Milestone 3A):
    #   1. Attach to `shm_img_name` and `shm_out_name` via `shared_memory.SharedMemory(name=...)`
    #   2. Wrap buffers as NumPy arrays with `np.ndarray(shape, dtype=np.float64, buffer=shm.buf)`
    #   3. Call `convolve_row_range(img_view, kernel, out_view, r_start, r_end)`
    #   4. Always call `shm.close()` in a `finally:` block!
    raise NotImplementedError("Implement _mp_shm_worker")


def convolve_multiprocessing_shm(
    img: np.ndarray, kernel: np.ndarray, num_workers: int = 4
) -> np.ndarray:
    """Run row-chunked 5x5 convolution across `num_workers` processes using SharedMemory."""
    # TODO (Milestone 3A):
    #   1. Allocate `shm_img` and `shm_out` with `shared_memory.SharedMemory(create=True, size=img.nbytes)`
    #   2. Copy `img` into `shm_img` and zero-initialize `shm_out`
    #   3. Spawn `num_workers` `Process` objects targeting `_mp_shm_worker`
    #   4. Start and join all processes, copy result out of `shm_out`, and clean up
    #      (`close()` AND `unlink()`) in a `finally:` block!
    raise NotImplementedError("Implement convolve_multiprocessing_shm")


# ==============================================================================
# MILESTONE 3B (25 pts): Native OpenMP-Style SMP with Numba (`prange`)
# ==============================================================================
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def convolve_numba(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Parallel 5x5 convolution using Numba `prange` across rows."""
        H, W = img.shape
        out = np.zeros((H, W), dtype=np.float64)
        # TODO (Milestone 3B): Parallelize the outer row loop using `prange(2, H - 2)`
        for r in range(2, H - 2):
            for c in range(2, W - 2):
                acc = 0.0
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        acc += img[r + dr, c + dc] * kernel[dr + 2, dc + 2]
                out[r, c] = acc
        return out

    @njit(parallel=True)
    def image_energy_numba(out: np.ndarray) -> float:
        """Parallel reduction computing sum_{r,c} out[r, c]^2."""
        H, W = out.shape
        total = 0.0
        # TODO (Milestone 3B): Parallelize with `prange(H)` (Numba auto-reduces `total += ...`)
        for r in range(H):
            for c in range(W):
                total += out[r, c] * out[r, c]
        return total


# ==============================================================================
# OPTIONAL BONUS (+15 pts): Non-Obvious Parallelism (256-Bin Brightness Histogram)
# ==============================================================================
# Problem: Count how many pixels fall into each brightness bin:
#     b = max(0, min(255, int(img[r, c] * 255.0)))
# Unlike convolution, write destinations `hist[b]` depend on pixel values and
# collide across workers! Implement and compare 3 multi-process versions below:
#   1. `histogram_mp_broken`  (5 pts): Shared `hist[256]` WITHOUT locks -> data race!
#   2. `histogram_mp_locked`  (5 pts): Shared `hist[256]` WITH `multiprocessing.Lock` -> exact, slow!
#   3. `histogram_mp_private` (5 pts): Per-worker private `local_hist[256]` merged at end -> exact, fast!

def histogram_sequential(img: np.ndarray) -> np.ndarray:
    """Reference single-core 256-bin brightness histogram."""
    hist = np.zeros(256, dtype=np.int64)
    H, W = img.shape
    for r in range(H):
        for c in range(W):
            b = max(0, min(255, int(img[r, c] * 255.0)))
            hist[b] += 1
    return hist


def histogram_mp_broken(img: np.ndarray, num_workers: int = 4) -> np.ndarray:
    """Optional Bonus Part 1 (5 pts): Unsynchronized shared `hist[256]` in SharedMemory.

    Spawn `num_workers` processes over row chunks [0, H) that increment a shared
    `hist[b] += 1` WITHOUT any lock. Observe that `hist.sum() < H * W` (lost updates!).
    """
    # TODO (Optional Bonus Part 1): Implement unsynchronized multi-process histogram
    raise NotImplementedError("Optional Bonus Part 1: Implement histogram_mp_broken")


def histogram_mp_locked(img: np.ndarray, num_workers: int = 4) -> np.ndarray:
    """Optional Bonus Part 2 (5 pts): Shared `hist[256]` protected by `multiprocessing.Lock`.

    Pass a `Lock()` to each worker so `with lock: hist[b] += 1` is atomic.
    Verify `np.array_equal(hist, ref_hist)` and measure the lock contention slowdown!
    """
    # TODO (Optional Bonus Part 2): Implement lock-protected multi-process histogram
    raise NotImplementedError("Optional Bonus Part 2: Implement histogram_mp_locked")


def histogram_mp_private(img: np.ndarray, num_workers: int = 4) -> np.ndarray:
    """Optional Bonus Part 3 (5 pts): Lock-free privatization across `num_workers` processes.

    Each worker allocates a private `local_hist = np.zeros(256, dtype=np.int64)`,
    counts its assigned rows with ZERO locks, and copies its 256-bin result into
    row `worker_id` of a SharedMemory array of shape `(num_workers, 256)`.
    The main process sums across `axis=0` after joining all workers.
    """
    # TODO (Optional Bonus Part 3): Implement lock-free privatized multi-process histogram
    raise NotImplementedError("Optional Bonus Part 3: Implement histogram_mp_private")


# ==============================================================================
# MAIN BENCHMARK & SELF-VERIFICATION HARNESS
# ==============================================================================
def main() -> None:
    kernel = get_kernels()["edges"]
    img = load_grayscale_image(DEFAULT_SMALL_IMAGE)
    H, W = img.shape
    print(f"Small image: {H}x{W}", flush=True)

    t0 = time.perf_counter()
    ref = convolve_sequential(img, kernel)
    t_seq = time.perf_counter() - t0
    print(f"Sequential Python: {t_seq:.6f} s", flush=True)

    chunked = np.zeros_like(img)
    for r0, r1 in make_row_chunks(4, 2, H - 2):
        convolve_row_range(img, kernel, chunked, r0, r1)
    assert np.allclose(ref, chunked), "Row-chunking mismatch"
    print("Row chunks: PASSED", flush=True)

    thread_times = {}
    for workers in [1, 4]:
        t0 = time.perf_counter()
        out = convolve_threads(img, kernel, workers)
        elapsed = time.perf_counter() - t0
        assert np.allclose(ref, out), "Thread output mismatch"
        thread_times[workers] = elapsed
        print(f"Python threads={workers}: {elapsed:.6f} s; "
              f"speedup vs Python={t_seq / elapsed:.2f}x", flush=True)
    print(f"Thread T1/T4: {thread_times[1] / thread_times[4]:.2f}x", flush=True)

    process_times = {}
    for workers in [1, 4]:
        t0 = time.perf_counter()
        out = convolve_multiprocessing_shm(img, kernel, workers)
        elapsed = time.perf_counter() - t0
        assert np.allclose(ref, out), "Process output mismatch"
        process_times[workers] = elapsed
        print(f"SharedMemory processes={workers}: {elapsed:.6f} s; "
              f"speedup vs Python={t_seq / elapsed:.2f}x", flush=True)
    print(f"Process T1/T4: {process_times[1] / process_times[4]:.2f}x", flush=True)

    if NUMBA_AVAILABLE:
        large = load_grayscale_image(DEFAULT_LARGE_IMAGE)
        for name, image in [("small", img), ("large", large)]:
            times = {}
            one_thread_out = None
            for workers in [1, 4]:
                set_num_threads(workers)
                warm = np.ascontiguousarray(image[:20, :20])
                warm_out = convolve_numba(warm, kernel)
                image_energy_numba(warm_out)

                t0 = time.perf_counter()
                out = convolve_numba(image, kernel)
                elapsed = time.perf_counter() - t0
                times[workers] = elapsed

                t0 = time.perf_counter()
                energy = image_energy_numba(out)
                energy_time = time.perf_counter() - t0
                assert np.isclose(energy, np.sum(out * out)), "Energy mismatch"
                assert np.all(out[:2, :] == 0) and np.all(out[-2:, :] == 0)
                assert np.all(out[:, :2] == 0) and np.all(out[:, -2:] == 0)
                if name == "small":
                    assert np.allclose(ref, out), "Numba small-image mismatch"
                if workers == 1:
                    one_thread_out = out
                else:
                    assert np.allclose(one_thread_out, out), "Numba thread-count mismatch"

                print(f"Numba {name}, threads={workers}: {elapsed:.6f} s; "
                      f"energy={energy:.8f}; reduction={energy_time:.6f} s", flush=True)
            print(f"Numba {name} T1/T4: {times[1] / times[4]:.2f}x", flush=True)
            if name == "small":
                plot_comparison(img, out, title="5x5 Edge Detection", save_path="result.png")
    else:
        print("Install Numba in this environment to complete Milestone 3B.")
        plot_comparison(img, ref, title="5x5 Edge Detection", save_path="result.png")

    try:
        t0 = time.perf_counter()
        ref_hist = histogram_sequential(img)
        print(f"Histogram sequential: {time.perf_counter() - t0:.6f} s", flush=True)
        for name, function in [("broken", histogram_mp_broken),
                               ("locked", histogram_mp_locked),
                               ("private", histogram_mp_private)]:
            t0 = time.perf_counter()
            hist = function(img, 4)
            elapsed = time.perf_counter() - t0
            if name != "broken":
                assert np.array_equal(ref_hist, hist), "Histogram bin mismatch"
                assert hist.sum() == img.size, "Histogram count mismatch"
            print(f"Histogram {name}: {elapsed:.6f} s; "
                  f"count={hist.sum()}; expected={img.size}", flush=True)
    except NotImplementedError:
        print("Optional histogram bonus not completed; skipped.", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
