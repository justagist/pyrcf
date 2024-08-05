"""Defines agents that can be used to compute control commands to be sent to
a robot given a global plan (target/task). When using 'classical' methods,
an agent may just be a local planner + a controller (use the PlannerControllerAgent in
this case). When using machine-learned controller, this could be a policy that direcly
uses the global plan and outputs a robot command (implement an agent from MLAgentBase)."""

from .agent_base import AgentBase, DummyAgent
from .planner_controller_agent import PlannerControllerAgent
from .ml_agent_base import MLAgentBase
from .torchscript_agent_base import TorchScriptAgentBase
