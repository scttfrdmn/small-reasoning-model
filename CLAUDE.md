# CLAUDE.md — Instructions for AI Assistants

This file contains conventions and requirements for AI assistants (Claude Code and others)
working in this repository.

---

## Changelog Maintenance

**Every code change must be reflected in `CHANGELOG.md`.**

- Place new entries under `## [Unreleased]` in the appropriate category:
  `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
- Use present-tense imperative: "Add NKI attention kernel" not "Added" or "Adds"
- Be specific: include file paths and what changed, not just "updated X"
- When a release is cut, move `[Unreleased]` entries under the new version heading
  (e.g. `## [0.2.0] - 2026-MM-DD`) and add a fresh empty `[Unreleased]` block

Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

---

## Code Style

- Formatter: **black** (`uv run black .`), line length 100, target Python 3.11
- Run black before committing any Python file
- Comments: well (even over-) commented — explain WHY, not just what
- Every non-trivial decision, formula, or PyTorch op should have an inline comment

---

## Architecture Constraints (Non-Negotiable)

These are hardware constraints, not preferences. Violating them silently degrades
Trainium2 efficiency and breaks quantization layouts:

- All `d_model`, `ffn_intermediate`, `vocab_size` must be **multiples of 128**
- `head_dim` must be **exactly 128**
- `n_heads % n_kv_heads == 0` (integer GQA ratio)
- These are asserted in `ModelConfig.__post_init__` — do not remove those assertions

---

## Import Conventions

- Training scripts import from `model.architecture`, not relative paths
- All subpackages (`model/`, `tokenizer/`, `data/`, `training/`, `eval/`, `inference/`)
  have `__init__.py` files
- The project is installed in editable mode via `uv sync`; imports work from the repo root

---

## Project Tracking and Status

Work is tracked in GitHub — **do not create status files, TODO lists, or
work-in-progress notes anywhere in the repository.**

- **Issues / tasks**: https://github.com/scttfrdmn/small-reasoning-model/issues
- **Project board**: https://github.com/users/scttfrdmn/projects/39
- **Milestones**: Phase 0 (Pre-training) → Phase 1 (SFT) → Phase 2 (GRPO) → v0.2 (Deployment)
- **Specification and design decisions**: `small-reasoning-model-spec.md`
- **Hardware and cost planning**: `docs/hardware.md`
- **Architecture rationale**: `docs/architecture.md`
- **Training recipe details**: `docs/training.md`

### When to create or update GitHub issues

- **New bug found** → open an issue with the `bug` label; include reproduction steps
- **New planned work** → open an issue with appropriate phase + domain labels;
  assign to the correct milestone
- **Work completed** → close the issue (do not leave it open as a "done" marker)
- **Work blocked** → add the `blocked` label and note what it's waiting on in a comment

### Labels

Domain: `model` `training` `data` `eval` `inference` `infrastructure`
Phase: `phase-0` `phase-1` `phase-2`
Status: `bug` `enhancement` `blocked`

### Do not

- Add task lists or `## TODO` sections to source files
- Create `STATUS.md`, `TASKS.md`, `PLAN.md`, or similar tracking files
- Use `CHANGELOG.md → Planned` for new work (GitHub issues are the canonical backlog)
- Leave "TODO: implement X" stubs without a corresponding open issue

---

## Running the Project

```bash
uv sync                        # install / sync dependencies
uv run black .                 # format all Python
uv run pytest                  # run tests
uv run srm-shape               # validate architecture (no GPU needed)
uv run srm-tokenizer --mode sample   # quick tokenizer smoke test
uv run srm-pretrain --config 500m --mode validate  # training loop smoke test
```

---

## Versioning

- `pyproject.toml` `version` field is the authoritative version
- Bump version in `pyproject.toml` when cutting a release
- Tag releases as `vMAJOR.MINOR.PATCH` (e.g. `v0.2.0`)
- Pre-1.0: minor bumps (`0.x.0`) for new capabilities, patch bumps (`0.x.y`) for fixes
