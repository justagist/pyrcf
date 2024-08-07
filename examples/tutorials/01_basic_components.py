"""Simple dummy control loop implementation demonstrating a typical control loop pipeline.

A typical control loop has these main components implemented from base classes:
    - RobotInterface
    - StateEstimatorBase
    - ControllerBase
    - LocalPlannerBase
    - GlobalMotionPlannerBase

See README.md to know exactly what each of these components do in the context of the PyRCF
framework. Also see the docs in the base classes to understand the mandatory functions to
be overridden when implementing these components (also see additional functions that can
be overridden if required).

This demo also demonstrates the contract between each components in the control loop.
Components are called in sequence in the control loop (no parallelisation -- see README
in this folder (examples/tutorials/README.md)).

Each component in the control loop only has access to specific information from
the robot and the previous component, in the form of well-defined data types.
This ensures that one component cannot otherwise influence the behaviour of
another component other than through these pre-defined interfaces. This also
allows defining each component independent of each other, making it easier to
define new components; they only have to follow the protocol of their corresponding
base class.
The most commonly used ones for each have already been implemented in PyRCF. You can design
your own components as long as they implement some minimal functionalities required by the
base classes of each component.

NOTE: You do not have to implement a CtrlLoop class yourself. It is recommended
that you use existing ones (such as in example 02). This example is purely for
conveying the concepts.

This is a dummy control loop where each component simply prints something to the screen.
"""

import time
from pyrcf.components.controllers import DummyController, ControllerBase
from pyrcf.components.state_estimators import StateEstimatorBase, DummyStateEstimator
from pyrcf.components.local_planners import DummyLocalPlanner, LocalPlannerBase
from pyrcf.components.global_planners import GlobalMotionPlannerBase
from pyrcf.components.robot_interfaces import DummyRobot, RobotInterface
from pyrcf.components.global_planners.ui_reference_generators import DummyUI

# pylint: disable=W0621, W0105

"""NOTE: Read the docs in the __main__ below before going through the implementation
of DummyCtrlLoop."""


class DummyCtrlLoop:
    """This class shows the basics of the MinimalCtrlLoop used in example 02.
    This is to show the major components and operation of a typical control loop.

    NOTE: You do not have to implement a CtrlLoop class yourself. It is recommended
    that you use existing ones (such as in example 03). This example is purely for
    conveying the concepts.
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        state_estimator: StateEstimatorBase,
        controller: ControllerBase,
        local_planner: LocalPlannerBase,
        global_planner: GlobalMotionPlannerBase,
    ):
        self.robot = robot_interface
        self.state_estimator = state_estimator
        self.controller = controller
        self.planner = local_planner
        self.global_planner = global_planner

    def run(self, loop_rate: float):
        """The main control loop.

        A typical control loop follows this structure, where each component is
        called in sequence with the output of the previous component.

        Args:
            loop_rate (float): loop rate in hz
        """
        start_t: float = time.time()
        prev_t: float = time.time() - start_t
        period: float = 1.0 / loop_rate
        while True:
            curr_t: float = time.time() - start_t
            dt: float = curr_t - prev_t

            # read latest robot state
            robot_state = self.robot.read()

            # use state estimator to compute robot states that are not directly observable from the
            # robot interface such as robot pose in the world, base velocity, foot contact states,
            # etc.
            robot_state = self.state_estimator.update_robot_state_with_state_estimates(
                robot_state=robot_state
            )

            # generate global plan for local planner
            global_plan = self.global_planner.generate_global_plan(
                robot_state=robot_state, t=curr_t, dt=dt
            )

            # generate local plan/references for controller using global plan target
            local_plan = self.planner.generate_local_plan(
                robot_state=robot_state, global_plan=global_plan, t=curr_t, dt=dt
            )

            # use the local plan to generate instantaneous command to be sent to the robot
            cmd = self.controller.update(
                robot_state=robot_state, local_plan=local_plan, t=curr_t, dt=dt
            )

            # write the commands to the robot
            self.robot.write(cmd=cmd)

            prev_t = curr_t
            # maintain control loop frequency (this is naive implementation)
            time.sleep(period)

            print()  # just for adding a gap in the stdout after each loop during prints


if __name__ == "__main__":
    # All of these components simply prints stuff out, but they have implemented the required
    # methods as required by their respective base classes, and hence can be used in the
    # control loop.

    # robot interface reads directly observable robot states and can write commands to actuators
    robot = DummyRobot()

    # state estimator is used when all states of the robot is not directly measurable, and some
    # need to be estimated using specific state estimation algorithms
    state_estimator = DummyStateEstimator()

    # a global planner is the component that gives a reference/goal/target for the robot to
    # achieve. This could be from a a high-level task planner, or (for testing control strategies
    # more interactively) from a keyboard/joystick (or any other such user interface). Such
    # user interfaces should inherit from UIBase class (which inherits from GlobalMotionPlannerBase
    # anyway).
    ui_as_global_planner = DummyUI()
    # although `ui_as_global_planner` (DummyUI) is an implementation of UIBase class, it is also
    # a valid global planner as DummyUI implements a UIBase, which inherits from GlobalPlannerBase.
    assert isinstance(ui_as_global_planner, GlobalMotionPlannerBase)

    # This is mostly a decoupling of what typically could be seen as part of 'controller'.
    # See 'Agent' definition in 'additional components' section in the README in this folder.
    # Local planner is treated an intermediary between the global planner and controller in case
    # this is needed. This can be useful in case of legged robots for instance,
    # where there is a need to have specific gait plans etc. to be able to follow a global plan.
    local_planner = DummyLocalPlanner()

    # controller computes the joint commands to be sent to the robot, given the robot state and
    # local plan message from local planner.
    controller = DummyController()

    # create a control loop object with these components and run at the required rate
    control_loop = DummyCtrlLoop(
        robot_interface=robot,
        state_estimator=state_estimator,
        controller=controller,
        local_planner=local_planner,
        global_planner=ui_as_global_planner,
    )

    ctrl_loop_rate: float = 50  # 50hz control loop

    try:
        control_loop.run(loop_rate=ctrl_loop_rate)
    except KeyboardInterrupt:
        print("Closing control loop")
