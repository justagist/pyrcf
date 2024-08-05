# TODO

## core

- [ ] base classes
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
  - [ ] debug datatypes
- [x] controller manager
- [x] control loop
- [x] logging
- [x] loop debuggers

## robot interfaces

- [x] dummy robot
- [x] bullet robot

## controllers

- [x] dummy controller

## local planners

- [x] dummy local planner

## UI/Global planners

- [x] dummy gp

## agents

- [x] dummy agent
- [x] planner controller agent
- [x] ml agent
- [x] pytorch agent

## controller managers

- [x] simple cm

## control loop implementations

- [x] simple managed cl
- [ ] minimal

## utilities

- [ ] filters
- [ ] frame_transforms
- [x] math_utils
- [x] urdf_utils
- [x] kd
    - [x] pinocchio
- [ ] data io

## examples

- [x] Dummy loop with core components
- [ ] robot loader
- [x] robot visualiser
- [ ]

# QoL

- [ ] add mjcf support to PybulletRobot
- [ ] use nptyping
- [ ] default np values in dataclass field not supported in python 3.11+
- [ ] exceptions
