"""A Python Robot Control Framework for quickly prototyping control algorithms for different robot
embodiments.

Primarily, this library provides an implementation of a typical control loop (via a
`MinimalCtrlLoop` (extended from `SimpleManagedCtrlLoop`) class), and defines interfaces for the
components in a control loop that can be used directly in these control loop implementations. It
also provides utility and debugging tools that will be useful for developing controllers and
planners for different robots. This package also provides implementations of basic controllers
and planners.

In the long run, this package will also provide implementations of popular motion planners and
controllers from literature and using existing libraries.

The names most commonly needed to build a control loop are re-exported here, so that a typical
script only needs a single import:

    >>> from pyrcf import MinimalCtrlLoop, PybulletRobot, JointPDController

Everything remains importable from its defining submodule as well (e.g.
`pyrcf.components.controllers.JointPDController`), which is where the less commonly used
components live.
"""

from .core.exceptions import (
    CtrlLoopExitSignal,
    NotConnectedError,
    PyRCFExceptionBase,
    UIException,
)
from .core.logging import PYRCF_LOGGER_NAME, logger, throttled_logging
from .core.types import (
    ControlMode,
    EndEffectorStates,
    GlobalMotionPlan,
    JointStates,
    LocalMotionPlan,
    PlannerMode,
    Pose3D,
    QuatType,
    RobotCmd,
    RobotState,
    StateEstimates,
    Twist,
    Vector3D,
)

# --- control loops ---
from .control_loop import MinimalCtrlLoop, SimpleManagedCtrlLoop

# --- component interfaces (implement these to add your own component) ---
from .components.agents import AgentBase, MLAgentBase, PlannerControllerAgent
from .components.controllers import ControllerBase
from .components.global_planners.global_planner_base import GlobalMotionPlannerBase
from .components.local_planners import LocalPlannerBase
from .components.robot_interfaces import RobotInterface
from .components.state_estimators import StateEstimatorBase
from .components.ctrl_loop_debuggers import CtrlLoopDebuggerBase
from .components.callback_handlers.base_callbacks import CustomCallbackBase
from .components.pyrcf_component import PyRCFComponent

# --- ready-to-use implementations ---
from .components.controllers import (
    GravityCompensatedPDController,
    JointPDController,
    SegwayPIDBalanceController,
)
from .components.local_planners import (
    BlindForwardingPlanner,
    JointReferenceInterpolator,
)
from .components.robot_interfaces.simulation import (
    MujocoRobot,
    PybulletRobot,
    SimulatedRobotInterface,
)
from .components.state_estimators import DummyStateEstimator
from .components.controller_manager import ControllerManagerBase, SimpleControllerManager
from .components.controller_manager.command_accumulators.cmd_accumulation_policies import (
    CmdMuxer,
    CommandAccumulatorBase,
    SimpleCmdOverride,
)

# --- clocks and rate helpers ---
from .utils.time_utils import (
    ClockBase,
    PythonEpochClock,
    PythonPerfClock,
    RateLimiter,
    RateTrigger,
)

__all__ = [
    # control loops
    "MinimalCtrlLoop",
    "SimpleManagedCtrlLoop",
    # component interfaces
    "AgentBase",
    "ControllerBase",
    "CtrlLoopDebuggerBase",
    "CustomCallbackBase",
    "GlobalMotionPlannerBase",
    "LocalPlannerBase",
    "MLAgentBase",
    "PyRCFComponent",
    "RobotInterface",
    "StateEstimatorBase",
    # implementations
    "BlindForwardingPlanner",
    "CmdMuxer",
    "CommandAccumulatorBase",
    "ControllerManagerBase",
    "DummyStateEstimator",
    "GravityCompensatedPDController",
    "JointPDController",
    "JointReferenceInterpolator",
    "MujocoRobot",
    "PlannerControllerAgent",
    "PybulletRobot",
    "SimulatedRobotInterface",
    "SegwayPIDBalanceController",
    "SimpleCmdOverride",
    "SimpleControllerManager",
    # datatypes
    "ControlMode",
    "EndEffectorStates",
    "GlobalMotionPlan",
    "JointStates",
    "LocalMotionPlan",
    "PlannerMode",
    "Pose3D",
    "QuatType",
    "RobotCmd",
    "RobotState",
    "StateEstimates",
    "Twist",
    "Vector3D",
    # exceptions
    "CtrlLoopExitSignal",
    "NotConnectedError",
    "PyRCFExceptionBase",
    "UIException",
    # logging
    "PYRCF_LOGGER_NAME",
    "logger",
    "throttled_logging",
    # clocks / rates
    "ClockBase",
    "PythonEpochClock",
    "PythonPerfClock",
    "RateLimiter",
    "RateTrigger",
]
