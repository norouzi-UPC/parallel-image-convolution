# Parallel Image Convolution

Team: MohammadJavad, Aland, Hamed and Wenyuan.

## Computer and method

- Computer / CPU: To be completed.
- Operating system: To be completed.
- Python and Numba versions: To be completed.
- Number of measurements: To be completed.

The input images are 600x800 and 1600x2400. We use the supplied 5x5 edge kernel. The outer two-pixel frame stays zero. Numba is warmed up before timing. Process times include starting and joining the workers, shared-memory setup and cleanup.

Use measurements from the same computer for all methods in each comparison table. If reporting results from another computer, give it a separate table and identify that computer.

## Small-image results

| Method | Time (s) | Sequential Python time / this time |
| --- | --- | --- |
| Sequential Python | To be completed | 1.00 |
| Python threads: 1 | To be completed | To be completed |
| Python threads: 4 | To be completed | To be completed |
| SharedMemory processes: 1 | To be completed | To be completed |
| SharedMemory processes: 4 | To be completed | To be completed |
| Numba threads: 1 | To be completed | To be completed |
| Numba threads: 4 | To be completed | To be completed |

| Comparison | T1 / T4 |
| --- | --- |
| Python threads | To be completed |
| SharedMemory processes | To be completed |
| Numba, small image | To be completed |
| Numba, large image | To be completed |

## Large image: Numba

| Threads | Time (s) | Energy |
| --- | --- | --- |
| 1 | To be completed | To be completed |
| 4 | To be completed | To be completed |

## Explanation

Why do the convolution workers not need a lock? To be completed.

Why do Python threads give little speedup for these loops? To be completed.

How do processes and SharedMemory help? To be completed.

What does Numba change, and how does the energy reduction work? To be completed.

Speedup over pure Python includes the benefit of compiled code. We use one versus four Numba threads to discuss native parallel scaling.

## Checks

- Sequential row chunks match the baseline: To be completed.
- Thread, process and Numba outputs match the baseline: To be completed.
- One-thread and four-thread Numba outputs match on the large image: To be completed.
- Energy agrees with `np.sum(out * out)`: To be completed.
- The outer frame is zero and the image has no chunk seams: To be completed.
- Shared-memory cleanup: To be completed.

## Optional histogram bonus

Status: Not started.

If attempted, add sequential, broken, locked and private times and counts. Explain lost updates, lock contention and private worker histograms. Otherwise remove this section from the final report.
