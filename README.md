# Parallel Image Convolution

UPC scientific programming and HPC assignment.

Team: MohammadJavad, Aland, Hamed and Wenyuan.

## Work split

| Person | Part | Branch |
| --- | --- | --- |
| MohammadJavad | Row chunks, final checks and report | `feature/chunks-report` |
| Aland | Python threads and GIL comparison | `feature/threading` |
| Hamed | Multiprocessing and SharedMemory | `feature/shared-memory` |
| Wenyuan | Numba convolution and energy reduction | `feature/numba` |

## Run on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe starter_convolution.py
```
