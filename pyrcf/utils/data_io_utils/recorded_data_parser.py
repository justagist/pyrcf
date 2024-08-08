from typing import List, Mapping, Any, Literal
import functools
import pickle
import numpy as np
from numbers import Number

from ...core.logging import logging


def rgetattr(obj, attr, *args):
    """Recursive version of python's `getattr` method to get nested sub-object attributes.
    Also tries to deal with `__getitem__` calls (e.g. dictionary keys, or indices)!

    Also handles [:] situations. E.g. "state_estiamates.end_effector_states.ee_poses[0][:].position"
    will return positions of first end-effector (parsing `position` attribute of `Pose3D` across
    all time stamps).
    """

    def _getattr(obj, attr):
        # recursive getattr that also tries to deal with __getitem__ calls
        split_at_bracket_end = attr.split("]")
        if len(split_at_bracket_end) > 1:
            # UGLY: should probably switch to regex
            remaining = "]".join(split_at_bracket_end[1:])
            spl = attr.split("[")
            str_attr = spl[0]
            val = spl[1].split("]")[0]
            if val.isnumeric():
                val = int(val)
            n_obj = obj if str_attr == "" else getattr(obj, str_attr, *args)
            if val != ":":
                item = n_obj[val]
                if remaining == "":
                    return item
                return _getattr(item, remaining)

            if remaining == "":
                if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) == 1:
                    return n_obj[0]
                return n_obj
            return (
                [_getattr(i, remaining) for i in n_obj]
                if len(n_obj) > 1
                else _getattr(n_obj[0], remaining)
            )

        else:
            if isinstance(obj, (list, tuple)):
                return (
                    [getattr(o, attr, *args) for o in obj]
                    if len(obj) > 1
                    else getattr(obj[0], attr, *args)
                )
            if isinstance(obj, np.ndarray):
                return np.array(
                    [getattr(o, attr, *args) for o in obj]
                    if len(obj) > 1
                    else getattr(obj[0], attr, *args)
                )
            return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class ComponentDataRecorderDataParser:
    """A data parser for reading file created using `ComponentDataRecorderDebugger`."""

    def __init__(self, file_name: str, load_on_init: bool = True):
        """A data parser for reading file created using `ComponentDataRecorderDebugger`.

        Args:
            file_name (str): Path to the file to read.
            load_on_init (bool, optional): If True, will read file and load all data to
                memory during init; otherwise only when data is queried. Defaults to True.
        """

        self._filename = file_name
        self._data = None
        self._keys = None
        self._data_length = None
        if load_on_init:
            self.load_data()

    def load_data(self):
        self._data = {
            "t": [],
            "dt": [],
            "robot_state": [],
            "global_plan": [],
            "agent_outputs": [],
            "robot_cmd": [],
            "debug_data": [],
        }
        self._keys = list(self._data.keys())
        self._data_length = 0
        logging.info("Parsing data...")
        with open(self._filename, "rb") as f:
            while True:
                try:
                    data_buffer = pickle.load(f)
                    for data in data_buffer:
                        self._data["t"].append(data["t"])
                        self._data["dt"].append(data["dt"])
                        self._data["robot_state"].append(data["robot_state"])
                        self._data["global_plan"].append(data["global_plan"])
                        self._data["agent_outputs"].append(data["agent_outputs"])
                        self._data["robot_cmd"].append(data["robot_cmd"])
                        self._data["debug_data"].append(data["debug_data"])
                        self._data_length += 1
                except EOFError:
                    break
        logging.info(f"Data parsing complete. Loaded data of length {self._data_length}.")

    @property
    def num_datapoints(self):
        """Length of data points in the loaded data."""
        return self._data_length

    @property
    def key_names(self):
        return self._keys

    def get_all_data(self) -> Mapping[str, List[Any]]:
        """Get all data as dictionary of lists.

        Available keys: ["t", "dt", "robot_state", "global_plan", "agent_outputs", "robot_cmd",
        "debug_data"]

        e.g. `data['t']` will be a list of timesteps from all the control loop iterations that were
        recorded.

        Returns:
            Mapping[str, List[Any]]: Output data.
                Available keys: ["t", "dt", "robot_state", "global_plan", "agent_outputs",
                "robot_cmd", "debug_data"]
        """
        if self._data is None:
            logging.info("Data was not loaded from file. Loading now...")
            self.load_data()
        return self._data

    def get_all_data_for_key(
        self,
        key_name: Literal[
            "t",
            "dt",
            "robot_state",
            "global_plan",
            "agent_outputs",
            "robot_cmd",
            "debug_data",
        ],
        field_name: str = None,
        as_ndarray_if_possible: bool = True,
    ) -> List[Any] | np.ndarray:
        """Get all the data for a specified field for all the objects of a key in the data
        dictionary.

        E.g. get_all_data_for_key("robot_state","state_estimates.pose.position") will return a
        numpy array of all the state_estimate.pose.position values from the data. If objects are not
        numbers or numpy arrays, they are returned as a list. So,
        `get_all_data_for_key("robot_state", "state_estimates.pose")` will return a list of Pose3D
        objects.

        Also allows index and key access for valid attributes.
        e.g.: get_all_data_for_key("robot_state",
        "state_estimates.end_effector_states.ee_poses[0].position[0]") is a valid call to get the x
        values of the end-effector pose of the first end-effector in the end-effector state
        object's `ee_poses` attribute.

        Also handles [:] situations. E.g.
        "state_estiamates.end_effector_states.ee_poses[0][:].position" will return positions of
        first end-effector (parsing `position` attribute of `Pose3D` across all time stamps).

        Args:
            key_name (Literal[ "t", "dt", "robot_state", "global_plan", "agent_outputs",
                "robot_cmd", "debug_data"]): The key to look for in the dictionary.
            field_name (str, optional): Nested attribute string to retrieve for the data in the
                specified key value in the dictionary. Defaults to None.
            as_ndarray_if_possible (bool, optional): If the retrieved objects is a number or numpy
                array, this option will allow returning a numpy array of the combined values.
                Defaults to True.

        Returns:
            List[Any] | np.ndarray: List of retrieved attributes from all objects of the specified
                key from the loaded data.
        """
        data = self._data[key_name]

        if field_name is None:
            output_list = data
        else:
            output_list = []
            for obj in data:
                try:
                    output_list.append(rgetattr(obj, field_name))
                except Exception as exc:
                    raise AttributeError(
                        f"Error trying to retrive field {field_name} for key: {key_name}"
                    ) from exc

        if as_ndarray_if_possible and (
            isinstance(output_list[0], np.ndarray) or isinstance(output_list[0], Number)
        ):
            output_list = np.array(output_list)
        return output_list
