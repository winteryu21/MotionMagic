# AGENTS.md — MotionMagic Project Rules

## 1. Project Overview

A 2D defense game where players cast spells via MediaPipe hand gesture recognition + PyTorch CNN classification.
Tech stack: Python 3.11+ · MediaPipe · PyTorch · Pygame-CE. Resolution: 1920×1080.

## 2. Directory Structure

```
src/ai/          # AI gesture recognition (collector, preprocessor, dataset, model, trainer, recognizer, rule_validator, aim_tracker)
src/game/        # Pygame-CE game (app, settings, scenes/, entities/, systems/, ui/)
src/bridge/      # AI↔Game integration (gesture_event, camera_thread)
scripts/         # CLI tools (collect_data, train_model, evaluate_model)
tests/           # pytest tests
assets/          # Game assets (sprites, effects, maps, ui, sounds, fonts)
data/            # AI training data (raw, processed, splits) — gitignored
models/          # Trained .pth checkpoints
docs/            # Architecture, gesture catalog, playtest logs
```

Dependency direction: `game/ → bridge/ → ai/` (one-way only). Reverse imports are forbidden.

## 3. Commands

```bash
pip install -r requirements.txt   # Install deps
python -m src.game.app            # Run game
pytest                            # Run tests
black . && ruff check .           # Lint & format
```

## 4. Coding Conventions

### Style (Auto-enforced)

- PEP 8 · `black` (88 chars) · `ruff` · `.editorconfig` (UTF-8, LF, 4-space indent)
- Before commit: `black . && ruff check . && pytest`

### Naming (See CONTRIBUTING.md, team agreement)

| Target | Case | Example |
|--------|------|---------|
| Module | `snake_case` | `rule_validator.py` |
| Class | `PascalCase` | `GestureCNN` |
| Function/Method | `snake_case` | `extract_finger_states()` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_ENEMIES_PER_WAVE` |

### Mandatory Rules

- All functions must have **type hints** + `from __future__ import annotations`
- All public classes/functions must have **Google-style docstrings**
- **Dimension comments** after tensor ops: `# (B, C=3, L=21)`
- No magic numbers → extract to `settings.py` or module-level constants
- AI recognition failure → return `None` (not exceptions) + `logging`
- Game constants are centralized in `src/game/settings.py`

### Import Order

1. Standard library → 2. Third-party → 3. Local (`src.ai`, `src.game`, `src.bridge`)

## 5. Architecture Constraints

- `src/ai/` must NEVER import from `src/game/` or `pygame`.
- `src/game/` must NEVER import from `torch` or `mediapipe`.
- All cross-boundary communication goes through `src/bridge/gesture_event.GestureEvent`.
- Camera capture and CNN inference run in a background thread (`src/bridge/camera_thread.py`), not in the game loop.

## 6. Testing

- Framework: `pytest`. Test files live in `tests/`.
- Name pattern: `tests/test_<module>.py`
- New logic must include corresponding tests.
- Run: `pytest` (configured via `pyproject.toml`).

## 7. Git Commits (Conventional Commits)

```
<type>(<scope>): <description>
type: feat | fix | refactor | docs | test | data | asset | chore
scope: ai | game | bridge | assets | docs
```

## 8. Dependency Management

`pip` + `venv`. `requirements.txt` (min versions) + `requirements-lock.txt` (`pip freeze`).
After any package change, always commit `pip freeze > requirements-lock.txt`.
