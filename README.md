# Parallel Image Convolution

UPC scientific programming and HPC assignment.

Team: MohammadJavad, Aland, Hamed and Wenyuan.

The professor has allowed a group submission. The original assignment slides are in `assignment/`.

## Work split

| Person | Part | Branch |
| --- | --- | --- |
| MohammadJavad | Row chunks, final checks and report | `feature/chunks-report` |
| Aland | Python threads and GIL comparison | `feature/threading` |
| Hamed | Multiprocessing and SharedMemory | `feature/shared-memory` |
| Wenyuan | Numba convolution and energy reduction | `feature/numba` |

Each person edits their own functions in `starter_convolution.py` and their contribution note in `docs/contributions/`. Open a pull request when the part is ready.

## Run on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter_convolution.py
```

This is a starter. The unfinished functions raise `NotImplementedError`. The full run will work after the required parts have been completed and merged.

The sequential convolution, row-range calculation and sequential histogram are supplied. The benchmark driver compares one and four workers and warms Numba up before timing.

## Submission

- `starter_convolution.py`
- `report.md`
- `result.png`

Use measurements from our own computers in the report.
