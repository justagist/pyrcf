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

Coverage of the pure/offline layers is now reasonable (`core/types`, control loop, logging,
`math_utils`, `frame_transforms`, `PinocchioInterface`, the PD controllers, the joint reference
interpolator, the data recorder round trip, and the debugger rate gating).

Remaining gaps:

- [ ] UI interfaces (`KeyboardGlobalPlannerInterface`, `JoystickGlobalPlannerInterface`,
      `PybulletGUIGlobalPlannerInterface`) -- need pygame/`inputs` to be faked
- [ ] `PybulletIKReferenceInterpolator` (needs a running pybullet IK interface)
- [ ] the pybullet/mujoco robot interfaces beyond contract tests (need a simulator, so probably
      better as opt-in integration tests than in the default suite)
- [ ] the zmq publisher/subscriber pair
- [ ] `PybulletRobotVisualizer` / `PybulletDebugRobot` (GUI)

## Known issues / follow-ups

- [ ] `MujocoRobot`: the MuJoCo Menagerie MJCFs and the matching URDFs frequently disagree, which
      constrains the MuJoCo backend. Verified: joint names agree for `go2`, `g1`, `anymal_c` but
      not `iiwa14`, `panda`, `h1`; the legged MJCFs only load with `floating_base=True` (their
      `home` keyframe is sized for a free joint); and MuJoCo cannot load any `robot_descriptions`
      URDF at all because the mesh URIs (`package://`, relative paths) go unresolved. The last two
      are fixable in `mujoco_robot` (drop/resize keyframes when changing the base; resolve
      `package://` and set `meshdir`).
- [ ] No single PD gain set is stable in both simulators for `go2`
      (`examples/utils_demos/demo_sim_backend_swap.py` documents the measurements). Root cause is
      that `GravityCompensatedPDController` is not a stance controller: for a floating base the
      generalised gravity term ignores ground reaction forces. A contact-aware whole-body or
      stance controller would narrow this considerably.
- [ ] `pylint` in the pinned pixi environment (3.2.6) emits a benign `W0012 unknown-option-value`
      for `too-many-positional-arguments`, which only exists in pylint >= 3.3. Clears on the next
      lock bump.
