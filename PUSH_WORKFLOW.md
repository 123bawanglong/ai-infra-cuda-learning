# Push Workflow For AI Infra Portfolio

This document defines the workflow for pushing code to this portfolio repository.

The goal is not just to upload code. The goal is to make the repository useful for AI infrastructure job applications by keeping it clean, reproducible, and easy for interviewers to understand.

## Repository Goal

Repository:

```text
ai-infra-cuda-learning
```

Purpose:

- Show CUDA learning progress for AI infrastructure and LLM inference roles.
- Keep each kernel version understandable and reproducible.
- Document profiling results and optimization reasoning.
- Avoid pushing random practice files, binaries, or temporary reports.

## Standard Push Checklist

Before every push, follow this checklist.

### 1. Check Current State

```bash
cd ~/course1
git status --short --branch
```

Confirm:

- The branch is `main`.
- Only intended files are staged or modified.
- Random practice files, binaries, Nsight reports, and generated inputs are not staged.

### 2. Decide Whether A File Belongs In The Portfolio

Good files to commit:

- Source code under `src/`
- Documentation under `docs/`
- Small helper scripts under `scripts/`
- README updates
- `.gitignore` updates

Do not commit by default:

- Local exercise files in the repository root, such as `course*.cu`
- Compiled binaries, such as `softmax_shared`, `course3.1_ncu`, or `hello_world`
- Nsight Compute `.ncu-rep` files
- Large generated input/output text files
- Temporary screenshots unless they are intentionally curated for documentation

If a practice file becomes portfolio-worthy, copy or move it into a clear path first, for example:

```bash
cp course3.1.cu src/softmax_shared.cu
```

### 3. Use Clear Portfolio File Names

Prefer names that explain the optimization idea:

```text
src/softmax_shared.cu
src/softmax_warp.cu
src/softmax_warp_shared.cu
docs/profiling.md
scripts/gen_input.py
```

Avoid names that only make sense during a course:

```text
course3.1.cu
course3.1.1.cu
course3.1.3.cu
```

Course-style names can remain locally, but GitHub should use portfolio-style names.

### 4. Verify The Code

Compile each changed CUDA file before pushing.

Use the full CUDA path if `nvcc` is not in PATH:

```bash
/usr/local/cuda-12.8/bin/nvcc -lineinfo src/softmax_shared.cu -o softmax_shared
/usr/local/cuda-12.8/bin/nvcc -lineinfo src/softmax_warp.cu -o softmax_warp
/usr/local/cuda-12.8/bin/nvcc -lineinfo src/softmax_warp_shared.cu -o softmax_warp_shared
```

Run a small correctness test:

```bash
printf "2 3\n1 2 3\n4 5 6\n" | ./softmax_warp_shared
```

Expected softmax values after the timing line:

```text
0.0900306
0.244728
0.665241
0.0900306
0.244728
0.665241
```

For multiple versions, run the same input on each version and confirm the softmax outputs match.

### 5. Stage Only Intended Files

Do not use `git add .` unless the repository has been carefully checked.

Prefer explicit staging:

```bash
git add README.md docs/profiling.md scripts/gen_input.py src/softmax_shared.cu src/softmax_warp.cu src/softmax_warp_shared.cu
```

Then check again:

```bash
git status --short
```

The staged files should be exactly the files intended for the portfolio update.

### 6. Commit With A Meaningful Message

Use commit messages that describe the portfolio improvement:

```bash
git commit -m "add softmax optimization variants and profiling notes"
```

Good commit message examples:

```text
add warp-shuffle softmax kernel
document nsight compute profiling workflow
add multi-warp softmax reduction
update softmax performance comparison
```

Avoid vague messages:

```text
update
fix
test
aaa
```

### 7. Push

```bash
git push
```

The remote should use SSH:

```bash
git remote -v
```

Expected form:

```text
origin  git@github.com:123bawanglong/ai-infra-cuda-learning.git (fetch)
origin  git@github.com:123bawanglong/ai-infra-cuda-learning.git (push)
```

## SSH Setup Notes

This WSL environment uses SSH over port `443` for GitHub because the normal SSH port may hang under the current network/proxy setup.

Expected `~/.ssh/config`:

```text
Host github.com
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

Test GitHub SSH authentication:

```bash
ssh -T git@github.com
```

Expected successful output:

```text
Hi 123bawanglong! You've successfully authenticated, but GitHub does not provide shell access.
```

## After Pushing

Open the GitHub repository and check:

```text
https://github.com/123bawanglong/ai-infra-cuda-learning
```

Confirm:

- README renders correctly.
- New source files appear under `src/`.
- Documentation appears under `docs/`.
- No accidental binaries or temporary files were uploaded.

## Interview-Focused Standard

Every pushed update should help answer at least one of these questions:

- What CUDA concept did I learn?
- What performance bottleneck did I identify?
- What optimization did I implement?
- How can the result be reproduced?
- How does this relate to AI infrastructure or LLM inference?

If a change does not help answer any of these questions, keep it local until it is cleaned up.
