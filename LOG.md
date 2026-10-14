# Phase 1 log

Machine: MacBook, Apple M5 (10 cores), macOS (Darwin 25.6.0), conda 26.1.1, Homebrew 7.0.7.
Date: 2026-10-14.

Reference: [ENGR6010_Exam1](https://github.com/aidanjmartin/ENGR6010_Exam1) at commit `90150da`, read-only, cloned outside this repository.

## Task 1: repository and environments

| step | command | outcome |
| :--- | :--- | :--- |
| clone Exam 1, read-only | `git clone https://github.com/aidanjmartin/ENGR6010_Exam1 <scratchpad>` | success: commit `90150da` |
| GitHub CLI | `brew install gh` | success: gh 2.102.0 |
| read pinned upstream sources | `git clone <url>` then `git checkout <commit>`, outside this repository | `unitree_sdk2_python` `814556d`: `python_requires='>=3.8'`, `cyclonedds==0.10.2` (`setup.py`). `unitree_webrtc_connect` `e0abc57`: `requires-python = ">=3.8"`, version 2.2.0 (`pyproject.toml`) |
| resolve Mac environment | `conda create --dry-run --override-channels -c conda-forge python=3.11 ...` (osx-arm64) | success: versions pinned in `env/environment-mac.yml` |
| resolve Ubuntu environment | the same with `--platform linux-64` and `pytorch=*=cuda*`, `CONDA_OVERRIDE_CUDA=12.8` | success: pytorch 2.14.1 CUDA 12.9 build; versions pinned in `env/environment-ubuntu.yml` |

## Differences from Exam 1

| item | Exam 1 | Phase 1 | why |
| :--- | :--- | :--- | :--- |
| robot computer | lab desktop, Ubuntu 26.04 | laptop, Ubuntu 24.04, RTX 4000 Ada Laptop GPU | the laptop goes with the robot; setup is repeated there |
| environments | one per part: Q2 Python 3.11 (DDS), Q3 Python 3.10 (WebRTC) | one per machine, Python 3.11, both robot libraries together on Ubuntu | one robot interface imports both backends; both libraries declare Python >= 3.8, and CycloneDDS 0.10.2 was proven on 3.11 in Exam 1. The two libraries were never installed together in Exam 1. |
| environment files | `requirements.txt` per part | conda `environment-*.yml` per machine | project brief |
| PyTorch | pip wheel, `cu128` index | conda-forge `pytorch=2.14.1` CUDA 12.9 build | conda-forge only; Exam 1 already used conda-forge-only environments |
| `.gitignore` | Python template plus a custom section | the custom section only, plus SB3 `.zip` checkpoints, `models/`, `runs/` | SB3 saves checkpoints as `.zip` |
| `.env.example` | four variables | same four; the AES key is marked not needed on this firmware | Exam 1 found `data2 = 2` |
| upstream clones | `~/go2_deps/`, commits in Exam 1 `q2_sdk/LOG.md` | same location and commits for CycloneDDS and `unitree_sdk2_python`; `unitree_webrtc_connect` installed by pip at `e0abc57` | Exam 1 `q3_apps/requirements.txt` pins it the same way |

## Problems found

None yet.
