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

### Example of a generic control loop

```python

# NOTE: this is not meant to be a script that you can use directly. See it as more of a pseudocode
# for a control loop that can be written using the framework.

period: float = 1.0 / loop_rate
while True:
    # read latest robot state
    robot_state = robot.read()

    # use state estimator to compute robot states that are not directly observable from the robot interface
    # such as robot pose in the world, base velocity, foot contact states, etc.
    robot_state = state_estimator.update_robot_state_with_state_estimates(
        robot_state=robot_state
    )

    # generate global plan for local planner (here, we simply generate a fake plan using user input)
    # but in real world scenarious, this is typically a trajectory plan or action sequence for
    # achieving a specific task goal (e.g. trajectory from point A to B avoiding collision, or
    # sequence of actions/trajectories to achieve a manipulation action)
    global_plan = global_planner.generate_global_plan()

    # generate local plan/references for controller using a local planner, given global plan target
    local_plan = planner.generate_local_plan(
        robot_state=robot_state, global_plan=global_plan
    )

    # use the local plan to generate instantaneous command to be sent to the robot
    cmd = controller.update(robot_state=robot_state, local_plan=local_plan)

    # write the commands to the robot
    robot.write(cmd=cmd)

    # maintain control loop frequency (this is naive implementation)
    time.sleep(period)

```

Tutorials and more details about concepts will be provided soon in the [tutorials](examples/tutorials) folder.
