# Contributing Guidelines

Thank you for your interest in improving this collection! Community contributions keep this repository up to date with the fast-moving autonomous driving perception literature.

## What to Contribute

- **New papers**: peer-reviewed papers from top venues (CVPR, ICCV, ECCV, NeurIPS, ICLR, ICML, AAAI, CoRL, ICRA, IROS, and major journals such as TPAMI, IJCV, T-ITS, IEEE Sensors Journal, IEEE IoT Journal), as well as influential arXiv preprints.
- **New datasets, benchmarks, or toolboxes** related to 3D perception for autonomous driving.
- **Fixes**: broken links, wrong venues/years, typos, or miscategorized entries.

## Entry Format

Please follow the existing format so the list stays consistent:

```markdown
- **MethodName** — Venue Year | [Paper](https://arxiv.org/abs/xxxx.xxxxx) | [Code](https://github.com/org/repo) — one-line description
```

Guidelines:

- Prefer the **arXiv abstract page** (`/abs/`, not `/pdf/`) for paper links.
- Link to the **official code release** when available; omit `[Code]` if none exists.
- Mark recent works (current or previous year) with the 🔥 emoji.
- Insert the entry in **chronological order** within the most appropriate section.
- Keep the one-line description factual and concise (what the method does, not how good it is).

## How to Submit

1. **Pull Request (preferred)**: fork the repo, create a branch, edit `README.md`, and open a PR with a short description of what you added.
2. **Issue**: if a PR is too much hassle, just [open an issue](https://github.com/Fishsoup0/Autonomous-Driving-Perception/issues) with the paper title and link — we will add it.

## Review

PRs are typically reviewed within a week. We may adjust the placement or wording of entries for consistency.

Thanks again for contributing! ⭐
