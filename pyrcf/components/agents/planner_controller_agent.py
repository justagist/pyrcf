import copy
from typing import Tuple

from ...core.types import GlobalMotionPlan, RobotCmd, RobotState, LocalMotionPlan
from .agent_base import AgentBase
from ..local_planners import LocalPlannerBase
from ..controllers import ControllerBase


class PlannerControllerAgent(AgentBase):
    """A simple implementation of Agent using a local planner object and controller.

    This agent simply calls the local planner update and controller update steps in sequence.
    """

    def __init__(self, local_planner: LocalPlannerBase, controller: ControllerBase):
        """Constructor.

        A simple implementation of Agent using a local planner object and controller.
        This agent simply calls the local planner update and controller update steps in sequence.


        Args:
            local_planner (LocalPlannerBase): The local planner object to be used in the loop.
            controller (ControllerBase): The controller to be used.
        """
        self.local_planner = local_planner
        self.controller = controller

        self._latest_local_plan = None
        self._latest_ctrl_cmd = None

    def get_action(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float = None,
        dt: float = None,
    ) -> RobotCmd:
        self._latest_local_plan = self.local_planner.generate_local_plan(
            robot_state=robot_state, global_plan=global_plan, t=t, dt=dt
        )
        self._latest_ctrl_cmd = self.controller.update(
            robot_state=robot_state,
            local_plan=self._latest_local_plan,
            t=t,
            dt=dt,
        )
        return self._latest_ctrl_cmd

    def get_latest_local_plan(self) -> LocalMotionPlan:
        return copy.deepcopy(self._latest_local_plan)

    def get_latest_ctrl_cmd(self) -> RobotCmd:
        return copy.deepcopy(self._latest_ctrl_cmd)

    def get_last_output(self) -> Tuple[LocalMotionPlan, RobotCmd]:
        return self.get_latest_local_plan(), self.get_latest_ctrl_cmd()

    def shutdown(self):
        self.controller.shutdown()
        self.local_planner.shutdown()

    def get_class_info(self) -> str:
        info = f"{self.get_class_name()}:\n"
        info += f"\t\t Controller: {self.controller.get_class_info()}\n"
        info += f"\t\t Local Planner: {self.local_planner.get_class_info()}"
        return info
