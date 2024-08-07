"""This demo does (almost) exactly the same thing as the previous example (01), but
uses the SimpleManagedCtrlLoop that is provided by PyRCF.

This is a dummy control loop where each component simply prints something to the screen.
"""

from pyrcf.components.robot_interfaces import DummyRobot
from pyrcf.components.agents import PlannerControllerAgent
from pyrcf.components.state_estimators import DummyStateEstimator
from pyrcf.components.controller_manager import SimpleControllerManager
from pyrcf.components.global_planners.ui_reference_generators import DummyUI
from pyrcf.components.controllers import DummyController
from pyrcf.components.local_planners import DummyLocalPlanner
from pyrcf.control_loop import SimpleManagedCtrlLoop

if __name__ == "__main__":
    # Load the same components as in the previous example.
    # All these components simply prints stuff out, but they have implemented the required
    # methods as required by their respective base classes, and hence can be used in the
    # control loop.
    robot = DummyRobot()
    state_estimator = DummyStateEstimator()
    controller = DummyController()
    local_planner = DummyLocalPlanner()
    global_planner = DummyUI()

    # create a controller manager to manage the controllers in the control loop. This is to
    # deal with cases where there are multiple controllers in the control loop. In our case
    # we don't necessarily need it, but this is required for creating the ControlLoop object.
    controller_manager = SimpleControllerManager()

    # the simple controller manager manages different 'agents'. In our case, our 'agent' is
    # just the local [planner and controller], so we use the PlannerControllerAgent implementation
    # of AgentBase class. This could be any control policy that can take a global plan message
    # and directly produce a control output to the robot without needing an intermediate local
    # planner (e.g. a machine-learned control policy (in which case you could look at the
    # MLAgentBase abstract class which is also derived from AgentBase)).
    agent = PlannerControllerAgent(local_planner=local_planner, controller=controller)

    # add this agent to the controller manager to handle its run in the control loop.
    controller_manager.add_agent(agent=agent)
    # NOTE: This could have been done during initialisation of the `SimpleControllerManager` object
    # by passing the list of agents to be used as an argument to the constructor. See doc in class
    # `SimpleControllerManager`.

    # create a control loop with these components.
    # The SimpleManagedCtrlLoop allows loading the different components and runs them at the
    # required rate. It also handles proper loading and shutting down of components, as well
    # as can provide some debugging and data logging capabilities.
    control_loop = SimpleManagedCtrlLoop(
        robot_interface=robot,
        state_estimator=state_estimator,
        controller_manager=controller_manager,
        global_planner=global_planner,
    )

    ctrl_loop_rate: float = 2  # 2hz control loop as an example

    # This will load the components in order (see log messages in terminal) and run them in sequence.
    control_loop.run(loop_rate=ctrl_loop_rate)

    # Sending a CtrlLoopExit signal (e.g. using KeyboardInterrupt) will close the control loop
    # cleanly (i.e. by shutting down each component cleanly. See log messages showing the shutdown
    # sequence.)
