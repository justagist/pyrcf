from abc import abstractmethod
from typing import Tuple, Literal
import numpy as np

from ...core.types import GlobalMotionPlan, RobotCmd, RobotState, JointStates, LocalMotionPlan
from .ml_agent_base import MLAgentBase
from ...core.logging import logging
from ...utils.time_utils import ClockBase, PythonPerfClock, RateTrigger

from ...variables import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch  # pylint: disable=E0401


class TorchScriptAgentBase(MLAgentBase):
    """A (abstract) base class that provides functionalities to use a torchscript model file
    and use it's inference mode to perform control update.

    Child class has to override two methods:
        - update_input_to_model
        - update_cmd_from_model_output
    See docstrings for each method in MLAgentBase class.
    """

    def __init__(
        self,
        model_file: str,
        input_dims: int,
        device: Literal["cpu", 0] = "cpu",
        warmup_iterations: int = 10,
        default_kp: np.ndarray | float = 20,
        default_kd: np.ndarray | float = 1.0,
        update_rate: float = None,
        clock: ClockBase = PythonPerfClock(),
        dtype: "torch.dtype" = None,
    ):
        """A (abstract) base class that provides functionalities to use a torchscript model file
        and use it's inference mode to perform control update.

        Child class has to override two methods:
            - update_input_to_model
            - update_cmd_from_model_output
        See docstrings for each method.

        Args:
            model_file (str): Path to torchscript (jit) file
            input_dims (int): Dimensions of the input vector to the model.
            device (Literal["cpu", 0] = "cpu", optional): Torch device to load tensors and model onto.
                Defaults to "cpu".
            warmup_iterations (int, optional): Number of iterations to send dummy input to the model
                on construction (for warming up model/processors). Defaults to 10.
            default_kp (np.ndarray | float, optional): Default joint positions gains. Defaults to 20.
            default_kd (np.ndarray | float, optional): Default joint damping gains. Defaults to 1.0.
            update_rate (float, optional): Update rate for the RL controller (if required to run at
                lower rate than the control loop). Use factors of the control loop's frequency.
                Defaults to None (run at the control loop rate).
            clock (ClockBase, optional): The clock to use for timer. Defaults to PythonPerfClock().
            dtype (torch.dtype, optional): Datatype of model parameters and tensors. Defaults to
                `torch.float`.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                f"Pytorch has to be installed for the {self.__class__.__name__} class to be used."
                " e.g. `pip install torch`."
            )

        if dtype is None:
            dtype = torch.float

        self._torch_device = torch.device(device)
        model = torch.jit.load(model_file, map_location=self._torch_device)
        model.eval()
        self._model = torch.jit.optimize_for_inference(model)

        self._device = device
        self._dtype = dtype
        self._dims = input_dims
        self._tensor_inp = torch.tensor(
            np.zeros(self._dims), dtype=self._dtype, device=self._torch_device
        ).reshape([1, -1])
        self._input_ndarray = np.zeros(self._dims)
        """The input array to be updated by the child class using the current robot
        state and global plan commands. This will be passed to the model as input during
        inference"""

        if warmup_iterations is not None and warmup_iterations > 0:
            logging.info("Warming up policy...")
            for _ in range(warmup_iterations):
                self._model.forward(self._tensor_inp)
            logging.info("Warm up complete.")

        self._latest_ctrl_cmd: RobotCmd = None
        """The RobotCmd object that can be use in the `update_input_to_model` method
        (if required) as the last sent command to the robot. This object has to be updated
        by the child class's `update_cmd_from_model_output` using the output from the model
        inference step."""

        self._default_kp = default_kp
        self._default_kd = default_kd

        self._rate = update_rate
        if self._rate is not None and self._rate > 0.0:
            self._rate_trigger = RateTrigger(rate=self._rate, clock=clock)
            logging.debug(f"{self.__class__.__name__}: Setting trigger rate to {self._rate}Hz.")

    def _should_run(self):
        if self._rate is None:
            return True
        if self._rate <= 0.0:
            return False
        return self._rate_trigger.triggered()

    def initialise_robot_cmd(self, joint_states: JointStates):
        self._latest_ctrl_cmd: RobotCmd = RobotCmd.fromJointStates(
            joint_states=joint_states,
            Kp=self._default_kp,
            Kd=self._default_kd,
        )
        self._latest_ctrl_cmd.joint_commands.joint_efforts *= 0.0
        self._latest_ctrl_cmd.joint_commands.joint_velocities *= 0.0

    @abstractmethod
    def update_input_to_model(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> np.ndarray:
        """Should update (`self._input_ndarray`) using appropriate values (input to model). This method
        has access to `self._latest_ctrl_cmd` (type `RobotCmd`) as well if needed.
        (NOTE: `self._latest_ctrl_cmd` is set to be the initial robot joint positions (zero velocities
        and efforts commands) with `default_kp` and `default_kd` at start).
        """

    @abstractmethod
    def update_cmd_from_model_output(
        self,
        model_output: np.ndarray,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float,
        dt: float,
    ) -> None:
        """Should update `self._latest_robot_cmd` (type `RobotCmd`) using the output from the NN model.

        Args:
            model_output (np.ndarray): the numpy array created from the output tensor from the model
                after the inference query was done. This is the output of the neural network. This
                method should use this object to update `self._latest_robot_cmd` to be sent to the
                robot.
        """

    def get_action(
        self,
        robot_state: RobotState,
        global_plan: GlobalMotionPlan,
        t: float = None,
        dt: float = None,
    ) -> RobotCmd:
        if self._latest_ctrl_cmd is None:
            self.initialise_robot_cmd(joint_states=robot_state.joint_states)
        # only perform inference and command update if required rate is met
        if self._should_run():
            # update the input tensor to the model by calling the child class's
            # `update_input_to_model` method.
            self.update_input_to_model(robot_state=robot_state, global_plan=global_plan, t=t, dt=dt)
            # update the pre-loaded tensor with the values from the updated
            # `self._input_ndarray` object.
            for idx in range(self._dims):
                self._tensor_inp[0, idx] = self._input_ndarray[idx]

            # perform inference using this input tensor
            with torch.inference_mode():
                output_tensor = self._model.forward(self._tensor_inp)

            # use the child class's `update_cmd_from_model_output` method
            # to update the RobotCmd object that is to be returned to the
            # control loop for writing to the robot
            self.update_cmd_from_model_output(
                model_output=output_tensor.cpu().detach().numpy().flatten(),
                robot_state=robot_state,
                global_plan=global_plan,
                t=t,
                dt=dt,
            )

        return self._latest_ctrl_cmd

    def get_last_output(self) -> Tuple[LocalMotionPlan, RobotCmd]:
        return None, self._latest_ctrl_cmd
