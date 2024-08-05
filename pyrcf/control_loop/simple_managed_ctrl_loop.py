from typing import List, Tuple

from ..core.logging import logging
from ..core.exceptions import CtrlLoopExitSignal
from ..components.robot_interfaces.robot_interface_base import RobotInterface
from ..components.controller_manager.controller_manager_base import ControllerManagerBase
from ..components.global_planners.global_planner_base import GlobalMotionPlannerBase
from ..components.state_estimators.state_estimator_base import StateEstimatorBase
from ..components.ctrl_loop_debuggers.ctrl_loop_debugger_base import CtrlLoopDebuggerBase
from ..components.callback_handlers.base_callbacks import CustomCallbackBase
from ..utils.time_utils import RateLimiter, PythonPerfClock, ClockBase


class _True_dt:

    def __init__(self):
        self._clock = PythonPerfClock()
        self._prev_t = self._clock.get_time()

    def get_dt(self):
        curr_t = self._clock.get_time()
        dt = curr_t - self._prev_t
        self._prev_t = curr_t
        return dt


class SimpleManagedCtrlLoop:
    """A minimal control loop that takes in a controller manager instead of a single controller and
    local planner.

    The run method runs the following in a loop of specified frequency.

    1. read robot state from robot interface
    2. update robot state with state estimates from state estimator
    3. get global plan message from global planner / user interface
    4. for agent in controller_manager.agents:
        a. compute control command to track the global plan (agent may be a Planner+Controller
            agent)
    5. accumulate commands from accumulation policy used by controller manager
    6. write control command to robot
    """

    def __init__(
        self,
        robot_interface: RobotInterface,
        state_estimator: StateEstimatorBase,
        controller_manager: ControllerManagerBase,
        global_planner: GlobalMotionPlannerBase,
        verbose: bool = True,
    ):
        """Create a minimal control loop where agents (controller+localplanner) are run in sequence.

        Args:
            robot_interface (RobotInterface): The robot interface to use in the loop.
            state_estimator (StateEstimatorBase): State estimator object to use in the loop.
            controller_manager (ControllerManagerBase): The controller manager object with agents
                pre-loaded.
            global_planner (GlobalMotionPlannerBase): The global planner object to be used in the
                loop.
        """
        self.robot = robot_interface
        self.state_estimator = state_estimator
        self.controller_manager = controller_manager
        self.global_planner = global_planner

        self._true_dt_getter = _True_dt()
        self._true_dt = 0.0
        self._loop_count = 0

        if verbose:
            msg = "\n----------------------------------------------------------------------------\n"
            msg += f"{self.__class__.__name__}: Control loop components:\n"
            msg += f"\tRobot: {self.robot.get_class_info()}\n"
            msg += f"\tGlobal Planner: {self.global_planner.get_class_info()}\n"
            msg += f"\tState Estimator: {self.state_estimator.get_class_info()}\n"
            msg += "\tAgents:\n"
            for n, agent in enumerate(self.controller_manager.agents):
                msg += f"\t\tAgent {n+1}: {agent.get_class_info()}\n"
            msg += "----------------------------------------------------------------------------\n"
            logging.info(msg=msg)

    def run(
        self,
        loop_rate: float,
        clock: ClockBase = PythonPerfClock(),
        debuggers: List[CtrlLoopDebuggerBase] = None,
        prestep_callbacks: List[CustomCallbackBase] = None,
        poststep_callbacks: List[CustomCallbackBase] = None,
    ):
        """The main control loop.

        Args:
            loop_rate (float): loop rate in hz. This will always use the system clock to try and
                maintain the desired control loop rate.
            clock (ClockBase, optional): Defines the clock to use for querying time for the
                components in the control loop. Time will be queried once per loop and the same `t`
                and `dt` will be passed to all components in that iteration. The option to select
                the clock is so that sim time can be used if required.
            debuggers (CtrlLoopDebuggerBase, optional): List of `CtrlLoopDebuggerBase` objects that
                are to be run in the control loop.
            prestep_callbacks (List[CustomCallbackBase], optional): List of `CustomCallbackBase` objects to
                be run in the control loop. Not recommended to do this except for debugging.
                Defaults to None.
            poststep_callbacks (List[CustomCallbackBase]. optional): List of additional `CustomCallbackBase`
                objects to be run in the control loop. Not recommended to do this except for
                debugging. Defaults to None.
        """
        if prestep_callbacks is None:
            prestep_callbacks = []

        if poststep_callbacks is None:
            poststep_callbacks = []

        for cb in prestep_callbacks + poststep_callbacks:
            assert isinstance(cb, CustomCallbackBase)

        logging.info(f"{self.__class__.__name__}: Activating robot...")
        while not self.robot.activate():
            continue
        logging.info(f"{self.__class__.__name__}: Robot activated.")

        rate = RateLimiter(frequency=loop_rate, name="control loop", clock=PythonPerfClock())
        start_t: float = clock.get_time()
        prev_t: float = clock.get_time() - start_t
        while True:
            try:
                curr_t: float = clock.get_time() - start_t
                dt: float = curr_t - prev_t

                self._loop_count += 1
                self._true_dt = self._true_dt_getter.get_dt()

                for cb in prestep_callbacks:
                    cb.run_once()

                # read latest robot state
                robot_state = self.robot.read()

                # use state estimator to compute robot states that are not directly observable from
                # the robot interface such as robot pose in the world, base velocity, foot contact
                # states, etc.
                robot_state = self.state_estimator.update_robot_state_with_state_estimates(
                    robot_state=robot_state, t=curr_t, dt=dt
                )

                # generate global plan for local planner
                global_plan = self.global_planner.generate_global_plan(
                    robot_state=robot_state, t=curr_t, dt=dt
                )

                # use the local plan to generate instantaneous command to be sent to the robot
                cmd = self.controller_manager.update(
                    robot_state=robot_state, global_plan=global_plan, t=curr_t, dt=dt
                )

                # write the commands to the robot
                self.robot.write(cmd=cmd)

                if debuggers is not None:
                    agent_outputs = [
                        agent.get_last_output() for agent in self.controller_manager.agents
                    ]
                    for debugger in debuggers:
                        debugger.run_once(
                            t=curr_t,
                            dt=dt,
                            robot_state=robot_state,
                            global_plan=global_plan,
                            agent_outputs=agent_outputs,
                            robot_cmd=cmd,
                        )

                for cb in poststep_callbacks:
                    cb.run_once()

                prev_t = curr_t

                rate.sleep()
            except (KeyboardInterrupt, CtrlLoopExitSignal):
                logging.info(
                    "Received control loop exit signal. Attempting shutdown of components..."
                )
                logging.info(f"{self.__class__.__name__}: Deactivating robot...")
                while not self.robot.deactivate():
                    continue
                logging.info(f"{self.__class__.__name__}: Robot deactivated.")
                self.shutdown()
                break

        if debuggers is not None:
            for debugger in debuggers:
                debugger.shutdown()

        for cb in prestep_callbacks + poststep_callbacks:
            cb.cleanup()

    def get_actual_loop_rate(self) -> Tuple[float, float]:
        """Get the true loop rate and dt (using system clock).

        Returns:
            Tuple[float, float]: True dt (s), true loop rate (hz)
        """
        return self._true_dt, (1 / self._true_dt) if self._true_dt > 0 else 0

    def get_loop_count(self) -> int:
        """Get the current number of iterations of the control loop.

        Returns:
            int: current number of iterations of the control loop
        """
        return self._loop_count

    def shutdown(self):
        """Cleanly shutdown all components in the control loop.
        Sequence of shutdown:
            - robot interface
            - controller manager
            - global planner
        """
        logging.info(f"{self.__class__.__name__}: Shutting down.")
        self.robot.shutdown()
        self.controller_manager.shutdown()
        self.global_planner.shutdown()
        logging.info(f"{self.__class__.__name__}: Control loop shut down complete.")

    def send_shutdown_signal(self):
        """Request clean shutdown of control loop and components."""
        raise CtrlLoopExitSignal
