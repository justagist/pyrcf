from pyrcf.components.robot_interfaces import DummyRobot
from pyrcf.components.agents import PlannerControllerAgent
from pyrcf.components.state_estimators import DummyStateEstimator
from pyrcf.components.controller_manager import SimpleControllerManager
from pyrcf.components.global_planners.ui_reference_generators import DummyUI
from pyrcf.components.global_planners import GlobalMotionPlannerBase
from pyrcf.components.controllers import DummyController
from pyrcf.components.local_planners import DummyLocalPlanner
from pyrcf.control_loop import SimpleManagedCtrlLoop

from pyrcf.components.callback_handlers import CustomCallbackBase


class PrintStuffCallback(CustomCallbackBase):

    def __init__(self, to_print: str):
        self._to_print = to_print

    def run_once(self) -> None:
        print(self._to_print)


if __name__ == "__main__":
    # All of these components simply prints stuff out, but they have implemented the required
    # methods as required by their respective base classes, and hence can be used in the
    # control loop.
    robot = DummyRobot()
    state_estimator = DummyStateEstimator()
    controller = DummyController()
    local_planner = DummyLocalPlanner()
    global_planner = DummyUI()

    # create a controller manager to manage the controllers in the control loop. This is to
    # deal with cases where there are multiple controllers in the control loop. In our case
    # we don't necessarily need it, but this is required for creating the Control loop instance.
    controller_manager = SimpleControllerManager()

    # the simple controller manager manages different 'agents'. In our case, our 'agent' is
    # just the local [planner and controller]
    agent = PlannerControllerAgent(local_planner=local_planner, controller=controller)

    # add this agent to the controller manager to handle it's run in the control loop.
    controller_manager.add_agent(agent=agent)
    # NOTE: This could have been done during initialisation of the `SimpleControllerManager` object
    # by passing the list of agents to be used as an argument to the constructor. See docs of
    # `SimpleControllerManager`.

    # although DummyUI is an implementation of UIBase class, it is also a valid global planner.
    assert isinstance(global_planner, GlobalMotionPlannerBase)

    # create a control loop with these components
    control_loop = SimpleManagedCtrlLoop(
        robot_interface=robot,
        state_estimator=state_estimator,
        controller_manager=controller_manager,
        global_planner=global_planner,
    )

    ctrl_loop_rate: float = 2  # 2hz control loop

    control_loop.run(
        loop_rate=ctrl_loop_rate,
        prestep_callbacks=[PrintStuffCallback(to_print="\nStarting a loop iteration...\n")],
        poststep_callbacks=[PrintStuffCallback(to_print="\nDone with loop iteration...\n\n")],
    )
