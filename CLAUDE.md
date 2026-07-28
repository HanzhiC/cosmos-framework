# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Start with AGENTS.md

**[`AGENTS.md`](./AGENTS.md)** is the canonical, actively-maintained repo map — commands, key file locations, and gotchas. Read it first; this file only adds Claude-Code-specific notes on top of it. Don't duplicate AGENTS.md here — if a command or path listed there changes, edit AGENTS.md, not this file.

Quick orientation (see AGENTS.md for the full tables):

- Everything lives in one Python package, `cosmos_framework/`. Training infra (data, model, trainer, callbacks, checkpoint, …) is top-level subpackages; inference (Diffusers/Transformers/vLLM-friendly core, Ray+Gradio serving) is isolated under `cosmos_framework/inference/`. Keep that separation — training-time imports don't belong in `inference/` and vice versa.
- Entry points are invoked as `python -m cosmos_framework.scripts.<name>` (`train.py`, `inference.py`, `export_model.py`, …). Training is driven by a pydantic-validated TOML recipe (`--sft-toml=<recipe>.toml`); schema at `cosmos_framework/configs/toml_config/sft_config.py`, recipe pattern in `examples/README.md`.
- Library-style shims that load Cosmos3 checkpoints into upstream ecosystems live under `packages/{transformers,vllm}-cosmos3/` (Diffusers needs no shim — diffusers ≥ 0.39 ships `Cosmos3OmniPipeline` natively).
- For a per-subpackage tour (including which subpackages are `[planned]` and not yet on disk), see [`docs/code_structure.md`](./docs/code_structure.md).

## Commands

Prefer the `just` recipes (they wrap `uv run` with the right sync group) over calling `uv`/`pytest` directly:

| Task | Command |
|---|---|
| Install | `just install` |
| Lint + format (auto-fix) | `just lint` |
| Type-check | `uv run pyrefly check` |
| Run all tests | `just test` |
| List tests | `just test-list` |
| Run a single test | `just test-single <test_name> [--pdb]` |
| Docker (CUDA 13.0 / 12.8) | `just docker-cu130` / `just docker-cu128` |

Test levels (`--levels`): 0 = smoke (≥1 GPU), 1 = partial E2E (≥8 GPUs), 2 = full E2E (≥8 GPUs). Test output/logs land in `outputs/pytest/<test_name>/{console,debug}.log`.

## Skills

Task-specific instructions live in `.agents/skills/` (canonical, mirrored to `.claude/skills/` for Claude Code): `cosmos3-setup`, `cosmos3-codebase-nav`, `cosmos3-inference`, `cosmos3-post-training`, `cosmos3-env-troubleshoot`. These are already registered as invokable skills — prefer invoking the matching one over re-deriving setup/inference/training/troubleshooting steps from scratch.

## Working conventions

- Always back non-trivial answers with `file:line` references rather than paraphrasing from memory.
- `uv.lock` is machine-generated — don't hand-edit it; let `uv sync` / `just install` regenerate it.
- Commits require DCO sign-off (`git commit -s`); see [`CONTRIBUTING.md`](./CONTRIBUTING.md).
