# Contributing

Development setup, code conventions, and workflow for contributors.

---

## Setup

```bash
git clone https://github.com/scottfriedman/small-reasoning-model
cd small-reasoning-model
uv sync            # creates .venv, installs all deps + dev deps, editable install
```

All commands below assume you are running inside the project with `uv run` or with the
`.venv` activated.

---

## Code Style

**Formatter:** [black](https://black.readthedocs.io/), line length 100.

```bash
uv run black .          # format all Python files
uv run black --check .  # check without modifying (CI use)
```

Black is configured in `pyproject.toml` (`[tool.black]`). Run it before every commit.
The project does not use isort or flake8 — black handles formatting; type hints are
encouraged but not enforced by a linter.

**Comment philosophy:** Well (even over-) commented. Every non-trivial line should explain
*why*, not just what. If you write a clever implementation, document why it's done that way.
If you use a specific numeric value, explain where it comes from. If you make a tradeoff,
name the alternatives you considered.

---

## Architecture Constraints

These are hardware constraints. Do not relax them:

- All `d_model`, `ffn_intermediate`, `vocab_size` must be **multiples of 128**
- `head_dim` must be **exactly 128**
- `n_heads % n_kv_heads == 0` (integer GQA ratio)

These are enforced by `ModelConfig.__post_init__`. If you add a new config, it must pass
these assertions. See [`docs/architecture.md`](architecture.md) for the rationale.

---

## Changelog

**Every meaningful change must be added to `CHANGELOG.md`** under `## [Unreleased]`.

Format: `[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)`. Categories:
`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

See [`CLAUDE.md`](../CLAUDE.md) for the full changelog maintenance policy.

---

## Versioning

[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

- **PATCH** (`0.x.Y`): bug fixes, documentation improvements, minor corrections
- **MINOR** (`0.X.0`): new capability (new module implemented, training phase complete, etc.)
- **MAJOR** (`X.0.0`): reserved for post-1.0 breaking changes

At pre-1.0, minor version bumps (`0.2.0`, `0.3.0`, ...) mark significant milestones:
- `0.2.0` — 500M validation pre-training complete
- `0.3.0` — 1B full pre-training complete
- `0.4.0` — SFT + GRPO complete, reasoning results available
- `1.0.0` — published, reproducible, documented results

Update `version` in `pyproject.toml` when cutting a release.
Tag releases as `vMAJOR.MINOR.PATCH`.

---

## Tests

```bash
uv run pytest               # run all tests
uv run pytest -x            # stop on first failure
uv run pytest tests/test_architecture.py  # specific file
```

Tests are in `tests/` (not yet created — add as you implement modules).

Minimum coverage expectations:
- `model/architecture.py`: shape tests for all three configs, forward pass smoke test
- `tokenizer/train_tokenizer.py`: digit isolation, round-trip fidelity (the verify() suite)
- `training/rewards.py`: math exact match, SymPy equivalence, format reward

---

## Stubs

Several modules are stubs (see `CHANGELOG.md` → Planned). When implementing a stub:

1. Replace the stub comment with real implementation
2. Add tests
3. Update `CHANGELOG.md` with a `### Added` or `### Changed` entry
4. Update `docs/` if the implementation differs from the spec

The stub files are in place so imports work and the project structure matches the spec
before every module is implemented. Do not remove stubs — replace them with real code.

---

## Running the Full Validation Chain

Before a training run, verify everything works end-to-end:

```bash
# 1. Architecture shape check (no GPU needed, ~1 second)
uv run srm-shape

# 2. Tokenizer smoke test
uv run srm-tokenizer --mode sample --output /tmp/tokenizer_test

# 3. Training loop smoke test (20 steps, synthetic data, ~30 seconds on GPU)
uv run srm-pretrain --config 500m --mode validate

# 4. Run tests
uv run pytest
```

All four must pass cleanly before starting a real training run.
