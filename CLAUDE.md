# CLAUDE.md

This file instructs Claude Code on how to work in this repository.

## Mission

You are helping prepare this codebase for a professional GitHub publication. Three tasks:

1. **Identify files to exclude** — anything unused, inappropriate for public release, or that should be gitignored (compiled artifacts, raw proprietary data, binary result blobs, academic PDFs under copyright, large media files).
2. **Write a README** — precise, technical, honest. Describe exactly what the code does based on what you read in the source. No filler, no marketing language.
3. **Assess cleanup necessity** — flag dead code, redundant notebooks, inconsistent naming, or anything that would embarrass a professional profile.

## Repository layout

```txt
homework3/
├── Dataset/
│   ├── raw/          ← raw CSVs (Capire March 2025 data + FF5 daily)
│   └── dataset.xlsx  ← processed dataset
├── articles/         ← reference PDFs
├── results/          ← .npz parameter files + .mp4 animations, by frequency
├── src/              ← core Python modules
│   ├── loader.py
│   ├── models.py
│   ├── trainer.py
│   └── plotter.py
├── update.py
├── main.ipynb
├── preprocess_dataset.ipynb
└── convergence_test.ipynb
```

## How to proceed

1. **Read all source files first**: `src/*.py`, `update.py`, all three notebooks. Do not guess at functionality from filenames.
2. **Read the results**: inspect `.npz` files (use `numpy.load`) to understand what parameters are saved and what the model outputs. Watch the `.mp4` filenames for clues about what is animated.
3. **Then and only then**: write the README and exclusion list.

## Exclusion candidates (verify before confirming)

- `__pycache__/` and `src/__pycache__/` — always exclude, add to `.gitignore`
- `articles/*.pdf` — likely copyrighted academic papers; exclude unless you have rights
- `Dataset/raw/*.csv` — may be proprietary (Capire data); exclude unless license permits redistribution
- `Dataset/dataset.xlsx` — derived from raw; exclude if raw is excluded
- `results/**/*.mp4` — large binary media; assess whether they add value or should be excluded
- `results/**/*.npz` — binary numpy archives; exclude unless they are meant to be reproducible reference outputs

## Output expected from you

### 1. `.gitignore`

Complete, no boilerplate padding. Only what this project actually needs.

### 2. `README.md`

Structure:

- One-paragraph abstract: what this code does, what data it uses, what it produces
- Requirements / installation
- How to run (exact commands, correct order)
- Output description (what files are produced, where)
- Data availability note (honest statement about whether data is included)

Tone: academic/technical. No emojis. No "feel free to". No "this project aims to".

### 3. Cleanup report

A flat list of specific actionable items. For each: what it is, why it's a problem, what to do. No vague suggestions.

## Constraints

- Do not modify any source file unless explicitly asked.
- Do not hallucinate functionality. If something is unclear after reading the code, say so explicitly.
- The GitHub profile is professional. Flag anything that looks like homework scaffolding, debug prints, hardcoded paths, or incomplete implementations.
