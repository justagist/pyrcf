# Changelog

## [0.0.7] - 2026-08-05

### Fixes

- `CmdMuxer` was completely non-functional: it returned `None`, passed a misspelled keyword to
  `RobotCmd.extend`, and cached its first result forever. Multi-agent control loops now work.
- `RobotCmd.extend()` raised `IndexError` on its default path because the gain arrays were not
  resized to match the extended joint list.
- `JointStates.extend()` and `EndEffectorStates.extend()` desynced their value arrays from their
  name lists when `overwrite_existing=True`.
- The control loop now shuts every component down on *any* exit path (via `try`/`finally`), not
  only on `KeyboardInterrupt`. Previously an error raised by a component (e.g. the segway
  controller's fall detection) left the simulator connected, input threads running and recorded
  debug data unflushed. A failure in one shutdown step no longer prevents the others.
- `SegwayPIDBalanceController` takeoff detection: `np.any(<generator>)` was always truthy, and the
  contact states were read from a non-existent attribute, so the air-return branch never ran.
- `SegwayPIDBalanceController` passed `low_pass_filter` arguments in the wrong order, which made
  the integrator decay a no-op and raised `AssertionError` on the ground-position filter.
- `throttled_logging` inspected its own frame instead of the caller's, so every call site in the
  codebase shared a single logger per level: whichever site fired first in a period permanently
  silenced the others, and records reported `logging.py` as their origin. Throttled loggers now
  also emit the first call immediately rather than swallowing it for a full period, and honour a
  changed delay for an existing call site.
- Removed shared mutable default arguments that leaked state between instances (control loop
  default planner/state estimator, UI `GlobalMotionPlan`, controller gains, numpy array and
  `pygame.Color` defaults). `flake8-bugbear` is now enabled in ruff to keep them out.
- `PyRCFTypesEncoder` silently encoded malformed debug objects as `null` instead of falling back
  to the standard json encoding.

### Changes

- pyrcf logs through its own `"pyrcf"` logger instead of reconfiguring the **root** logger at
  import time. Applications keep full control of their own logging setup. Use
  `from pyrcf import logger` (or `logging.getLogger("pyrcf")`) to change pyrcf's log level.
- The most commonly used names are re-exported from the top-level package, so a typical script
  needs a single `from pyrcf import ...`. Submodule imports continue to work unchanged.
- The package now ships a PEP 561 `py.typed` marker, plus PyPI classifiers and project URLs.
- `pyrcf-visualise-robot` uses `argparse` and supports `--help`. **Breaking:** the optional second
  positional argument is replaced by the `--floating-base` flag (its old meaning contradicted its
  documented name).
- CI now uses non-mutating lint tasks (`black --check`, `ruff check`) that can actually fail the
  build, and runs `pylint` (previously configured but never executed in CI). Coverage measures the
  whole `pyrcf` package rather than only imported modules.
- **Releases are now gated on pushing a version tag** (e.g. `v0.0.7`) instead of auto-publishing
  on every version bump pushed to `main`, matching the release workflow used by `pybullet_robot`
  and `roschema`. The workflow lints and tests the tagged commit (a tag push does not trigger
  `ci.yml`), checks that the tag matches `project.version` in `pyproject.toml`, builds and
  `twine check`s the artifacts, and publishes via PyPI Trusted Publishing from a `pypi`
  environment, so no PyPI token is stored in the repository any more.

### Adds

- Tests for the command accumulators, control loop shutdown lifecycle, logging, filters and the
  segway controller's takeoff handling. Package coverage went from 18% to 48%.
- Documentation for `SegwayPIDBalanceController` in the controllers README.

## [0.0.6] - 2024-08-10

### Adds

- Introduce a more complete framework
- Major bug fixes and refactoring
- Introduce several new implementations of controllers and planners
- Introduce several implementations of debuggers and callback handlers
- Improve documentation and README
- Add more examples and tutorials

## [0.0.4] - 2024-08-08

### Adds

- Example on using SimpleManagedCtrlLoop and core components
- Improve docs and Readme

## [0.0.3]

### Adds

- Auto-release action in main branch on version bump

### Fixes

- Pypi installation issues
- mutable defaults in dataclass fields

## [0.0.1]

### Adds

- Initial working version of framework
- Minimal dummy example of control loop
