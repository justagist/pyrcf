# pyrcf

**WORK IN PROGRESS. NOT READY FOR USE.**

A Python Robot Control Framework for quickly prototyping control algorithms for different robot embodiments.

Primarily, this library provides an implementation of a typical control loop (via a `MinimalCtrlLoop` (extended from `SimpleManagedCtrlLoop`) class),
and defines interfaces for the components in a control loop that can be used directly in these control loop implementations. It also provides utility and debugging tools that will be useful for developing controllers and planners for different robots. This package also provides implementations of basic
controllers and planners.

In the long run, this package will also provide implementations of popular motion planners and controllers from literature and using existing libraries.

## Continuous Integration Status

[![Ci](https://github.com/justagist/pyrcf/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/justagist/pyrcf/actions/workflows/ci.yml?query=branch%3Amain)
[![Codecov](https://codecov.io/gh/justagist/pyrcf/branch/main/graph/badge.svg?token=Y212GW1PG6)](https://codecov.io/gh/justagist/pyrcf)
[![GitHub issues](https://img.shields.io/github/issues/justagist/pyrcf.svg)](https://github.com/justagist/pyrcf/issues/)
[![GitHub pull-requests merged](https://badgen.net/github/merged-prs/justagist/pyrcf)](https://github.com/justagist/pyrcf/pulls?q=is%3Amerged)
<!-- [![GitHub release](https://img.shields.io/github/release/justagist/pyrcf.svg)](https://github.com/justagist/pyrcf/releases/) -->
[![License](https://img.shields.io/pypi/l/bencher)](https://opensource.org/license/mit/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/downloads/)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

## PyRCF Philosophy

PyRCF follows the principle of a single thread control loop where components are communicating with each other strictly using pre-defined message types,
and run sequentially.

### A generic control loop

```text

LOOP:
  # Read latest robot state
  robot_state = ROBOT_INTERFACE->read_robot_state()

  # Update robot state with estimations (when all states are not directly measurable)
  robot_state = STATE_ESTIMATOR->update_robot_state_estimates(robot_state)

  # Generate global plan (high-level task objective or target)
  global_plan = GLOBAL_PLANNER->generate_global_plan()

  # Generate local plan based on state and global plan
  local_plan = LOCAL_PLANNER->generate_local_plan(robot_state, global_plan)

  # Generate control command based on state and local plan
  cmd = CONTROLLER->compute_commands(robot_state, local_plan)

  # Send command to robot
  ROBOT_INTERFACE->write_robot_command(cmd)

  # Maintain loop frequency (naive implementation)
  SLEEP(period)

END LOOP

```

This package provides interfaces to define custom components (such as controller, robot interface, global planner,
local planner, etc) that can be run in a control loop, as well as provides an implementation of a control loop
class which can execute these components in the required order at the specified rate. Implementations of simple
forms of all components are available in this package, including simulated interfaces for many robot embodiments.

Custom controllers and planners can be implemented and quickly tested on existing robot interfaces or on custom
robot interfaces (which can be easily defined).

More complex algorithms for control and planning will be provided by this package over time.

Tutorials and more details about concepts will be provided soon in the [tutorials](examples/tutorials) folder.
