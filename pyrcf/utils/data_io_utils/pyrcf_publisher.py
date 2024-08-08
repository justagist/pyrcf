from abc import ABC, abstractmethod
import json
import dataclasses
import enum
import numpy as np
import zmq

# local imports for encoding custom data types
from ...core.types import Pose3D, JointStates, EndEffectorStates, RobotCmd
from ...core.types.debug_types import (
    ListElementsCompareType,
    JointStatesCompareType,
    Pose3DCompareType,
)
from ..math_utils import quat2rpy

DEFAULT_ZMQ_PUBLISH_PORT: int = 5001
DEFAULT_PLOTJUGGLER_PUBLISH_PORT: int = 9872


class PyRCFTypesEncoder(json.JSONEncoder):
    """Custom json encoding for handling datatypes in custom PyRCF datatypes."""

    def default(self, o):
        if isinstance(o, JointStatesCompareType):
            # custom formatting to make comparing joint states easier
            if (
                o.joint_states_list is not None
                and len(o.joint_states_list) > 0
                and o.joint_states_list[0].joint_names is not None
            ):
                jnames = o.joint_states_list[0].joint_names
                data = {
                    "joint_positions": {},
                    "joint_velocities": {},
                    "joint_efforts": {},
                }
                row_names = o.get_row_names()
                for jname in jnames:
                    data["joint_positions"][jname] = {}
                    data["joint_velocities"][jname] = {}
                    data["joint_efforts"][jname] = {}
                    for n, row in enumerate(row_names):
                        try:
                            p, v, t = o.joint_states_list[n].get_state_of(joint_name=jname)
                            if p is not None:
                                data["joint_positions"][jname][row] = p
                            if v is not None:
                                data["joint_velocities"][jname][row] = v
                            if t is not None:
                                data["joint_efforts"][jname][row] = t
                        except (KeyError, AttributeError):
                            continue
                return data
        elif isinstance(o, Pose3DCompareType):
            # custom formatting to make comparing Pose3D objects easier
            if o.pose_list is not None and len(o.pose_list) > 0:
                data = {
                    "position": {"x": {}, "y": {}, "z": {}},
                    "RPY (deg)": {"roll": {}, "pitch": {}, "yaw": {}},
                }
                row_names = o.get_row_names()
                for n, name in enumerate(row_names):
                    if o.pose_list[n] is None:
                        continue
                    data["position"]["x"][name] = o.pose_list[n].position[0]
                    data["position"]["y"][name] = o.pose_list[n].position[1]
                    data["position"]["z"][name] = o.pose_list[n].position[2]
                    rpy = np.rad2deg(quat2rpy(o.pose_list[n].orientation))
                    data["RPY (deg)"]["roll"][name] = rpy[0]
                    data["RPY (deg)"]["pitch"][name] = rpy[1]
                    data["RPY (deg)"]["yaw"][name] = rpy[2]
                return data
        elif isinstance(o, ListElementsCompareType):
            # custom formatting to make comparing two similar list of floats easier
            if o.check_format(raise_exception_if_wrong=False):
                rows = o.get_row_names()
                cols = o.elements
                vals = o.get_values()
                data = {}
                for c, col in enumerate(cols):
                    data[col] = {rows[i]: vals[i, c] for i in range(len(rows))}
                return data
        elif isinstance(o, Pose3D):
            # add rpy and quaternion for orientation data
            return {
                "position": o.position,
                "orientation": {
                    "quaternion": dict(zip(["x", "y", "z", "w"], o.orientation)),
                    "rpy (degrees)": dict(
                        zip(
                            ["roll", "pitch", "yaw"],
                            np.rad2deg(quat2rpy(o.orientation)),
                        )
                    ),
                },
            }
        elif isinstance(o, EndEffectorStates):
            # hack to make ee names show up for corresponding values
            data = {}
            _ee_names = o.ee_names
            for field in dataclasses.fields(o):
                if field.name == "ee_names":
                    continue
                attr_vals = getattr(o, field.name)
                if attr_vals is not None and field.name in [
                    "contact_states",
                    "contact_forces",
                    "ee_poses",
                    "ee_twists"
                ] and len(_ee_names) == len(attr_vals):  # fmt: skip
                    # expose ee values next to ee names
                    data[field.name] = dict(zip(_ee_names, self.default(attr_vals)))
            return data
        elif isinstance(o, RobotCmd) and o.joint_commands.joint_names is not None:
            _j_names = o.joint_commands.joint_names
            data = {}
            for field in dataclasses.fields(o):
                attr_vals = getattr(o, field.name)
                if field.name == "joint_commands":
                    data[field.name] = self.default(attr_vals)
                elif attr_vals is not None and len(_j_names) == len(attr_vals):
                    # expose joint values next to joint names
                    data[field.name] = dict(zip(_j_names, self.default(attr_vals)))
            return data
        elif isinstance(o, JointStates):
            # hack to make joint names show up for corresponding values
            data = {}
            _j_names = o.joint_names if o.joint_names is not None else []
            for field in dataclasses.fields(o):
                if field.name == "joint_names":
                    continue
                attr_vals = getattr(o, field.name)
                if (
                    attr_vals is not None
                    and field.name
                    in [
                        "joint_positions",
                        "joint_velocities",
                        "joint_efforts",
                    ]
                    and len(_j_names) == len(attr_vals)
                ):
                    # expose joint values next to joint names
                    data[field.name] = dict(zip(_j_names, self.default(attr_vals)))
                # else:
                #     data[field.name] = self.default(attr_vals)
            return data
        elif dataclasses.is_dataclass(o):
            # handling all other dataclasses (this takes care of all other PyRCF custom types)
            data = {}
            for field in dataclasses.fields(o):
                data[field.name] = self.default(getattr(o, field.name))
            return data
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, list):
            # handling weird issue when o = []
            return list(o)
        elif o is None:
            return ""
        elif isinstance(o, enum.Enum):
            # handling Enum types to show up with their int values (e.g. PlannerMode)
            return {"current value": o.value, o.name: o.value}
        elif isinstance(o, np.number):
            # handling weird issue of single numbers in np types (e.g. np.int64)
            return float(o)
        else:
            # for all other types use default encoding if available
            try:
                return json.JSONEncoder.default(self, o)
            except (TypeError, OverflowError) as e:
                # exposing culprit along with the original error
                raise TypeError(f"{e}: culprit: {o}") from e


class PyRCFPublisherBase(ABC):
    """Base publisher class."""

    @abstractmethod
    def publish(self, data: dict):
        """Publish serialised data using this PyRCFPublisher."""
        raise NotImplementedError("Should be implemented in child class.")

    def close(self):
        """Close publisher cleanly."""
        return


class PyRCFPublisherZMQ(PyRCFPublisherBase):
    """
    A zero MQ publisher for publishing json serialised data to specified port.
    """

    def __init__(self, port: int = DEFAULT_ZMQ_PUBLISH_PORT):
        """Constructor.
        A zero MQ publisher for publishing json serialised data to specified port.

        Args:
            port (int, optional): port to publish in. Defaults to 5001.
        """
        self.socket = zmq.Context().socket(zmq.PUB)
        self.socket.bind("tcp://*:" + str(port))

    def publish(self, data: dict):
        """Publish the provided dictionary data as a JSON string to the pre-defined port.

        Args:
            data (dict, optional): Data as a dictionary.
        """
        self.socket.send_string(json.dumps(data, cls=PyRCFTypesEncoder))

    def close(self):
        self.socket.close()

    def __del__(self):
        self.close()


class PlotJugglerPublisher(PyRCFPublisherZMQ):
    """
    PlotJugglerPublisher is used to stream data to PlotJuggler and plot them in real time.
    On PlotJuggler, choose ZMQ Subscriber for data streaming, the port is used to identify the data
    channel.
    Publishes data to tcp port 9872 (default for plotjuggler).
    """

    def __init__(self):
        """
        PlotJugglerPublisher is used to stream data to PlotJuggler and plot them in real time.
        On PlotJuggler, choose ZMQ Subscriber for data streaming, the port is used to identify the
        data channel.
        Publishes data to tcp port 9872 (default for plotjuggler).
        """
        super().__init__(port=DEFAULT_PLOTJUGGLER_PUBLISH_PORT)
