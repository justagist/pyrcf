# TODO

## core

- [x] base classes
  - [x] pyrcf_component
  - [x] controller
  - [x] robot interface
  - [x] global planner
  - [x] local planner
  - [x] agent
  - [x] loop debugger
  - [x] custom callbacks
- [ ] data types
  - [x] robot io
  - [x] tf types
  - [x] planner types
  - [x] motion datatypes
  - [x] debug datatypes
- [x] controller manager
- [x] control loop
- [x] logging
- [x] loop debuggers

## robot interfaces

- [x] dummy robot
- [x] bullet robot

## controllers

- [x] dummy controller
- [x] joint pd controller
- [x] gravity compensated joint pd controller
- [x] segway pid balance controller

## local planners

- [x] dummy local planner
- [x] blind forwarding planner
- [x] joint reference interpolator
- [x] pybullet ik reference interpolator

## UI/Global planners

- [x] dummy gp
- [x] keyboard interface
- [x] joystick interface
- [x] pybullet gui interface

## agents

- [x] dummy agent
- [x] planner controller agent
- [x] ml agent
- [x] pytorch agent

## controller managers

- [x] simple cm
- [ ] controller switching / starting / stopping

## control loop implementations

- [x] simple managed cl
- [x] minimal

## utilities

- [x] filters
- [x] frame_transforms
- [x] math_utils
- [x] urdf_utils
- [x] kd
    - [x] pinocchio
- [x] data io

## examples

- [x] Dummy loop with core components
- [x] robot visualiser
- [ ] robot loader

# QoL

- [ ] add mjcf support to PybulletRobot
- [ ] use nptyping
- [ ] default np values in dataclass field not supported in python 3.11+
- [ ] exceptions

## Testing

Test coverage is still low outside `core/types`, the control loop and the logging module.
Highest-value gaps, roughly in order:

- [ ] `PinocchioInterface` (largest untested module)
- [ ] `math_utils` / `frame_transforms` (pure functions, cheap to test)
- [ ] local planners (`JointReferenceInterpolator`, `PybulletIKReferenceInterpolator`)
- [ ] `JointPDController` / `GravityCompensatedPDController`
- [ ] debuggers, especially the record/parse round trip of `ComponentDataRecorderDebugger`
- [ ] UI interfaces (need pygame/joystick to be faked)

## Known issues / follow-ups

- [ ] `KeyboardGlobalPlannerInterface(parallel_mode=True)` starts a thread that processes the
      pygame event queue exactly once and then exits, so no input is recorded. Either loop the
      body or drop the option. (pygame events are also not safe to poll off the main thread.)
- [ ] `JoystickGlobalPlannerInterface` reader thread is non-daemon and blocks in `get_gamepad()`,
      so `shutdown()` can hang until the next gamepad event. `_check_connection()` likewise blocks
      until an event arrives, which makes `MinimalCtrlLoop.useWithDefaults()` hang when an idle
      gamepad is plugged in.
- [ ] `CtrlLoopDebuggerBase.run_once` deep-copies the whole loop state on every trigger, and
      agent outputs are deep-copied a second time by `PlannerControllerAgent.get_last_output()`.
      Costly at high loop rates.
- [ ] `importing pyrcf` pulls in pygame (via the UI interfaces imported by `MinimalCtrlLoop`),
      which prints a banner and initialises SDL. Consider importing those lazily.
- [ ] `PybulletRobot.read()` overwrites `ee_names` with pinocchio's ordering after populating
      `contact_states` from the simulator's ordering; add an assertion that the two agree.
- [ ] enable ruff's `B905` (`zip(..., strict=)`) once the affected call sites have tests.
- [ ] `pyrcf.utils.frame_transforms` imports `multiplyTransforms`/`invertTransform` from pybullet
      for pure quaternion maths that scipy (already a dependency) can do.
