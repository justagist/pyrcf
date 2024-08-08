"""
This file is purely for defining data types for easier debugging when publishing
data via zmq e.g. to plotjuggler.

These data types should make debugging easier for lists of elements, lists of
JointStates objects, and lists of Pose3D objects. This is done by making the
custom encoder in plotjuggler publisher handle these in special ways that makes
plotting data easier in plotjuggler.
"""

from dataclasses import dataclass
from typing import List
import numpy as np

from .robot_io import JointStates
from .tf_types import Pose3D


@dataclass
class ListElementsCompareType:
    """A custom data type defined to make comparison of elements in equal-sized lists easier.
    Use this type to publish data to plotjuggler using PlotjugglerPublisher, which will handle
    encoding of this data type to make it easier to compare elements in plotjuggler.
    """

    elements: List[str]
    """The name of elements in the list."""
    values: List[List[float]] | np.ndarray = None
    """The lists to compare."""
    row_names: List[str] = None
    """Optional name for the different lists."""

    def get_row_names(self) -> List[str]:
        if self.row_names is not None:
            return self.row_names
        else:
            return [f"data_{i+1}" for i in range(len(self.values))]

    def get_values(self) -> np.ndarray:
        return np.array(self.values)

    def check_format(self, raise_exception_if_wrong: bool = True) -> bool:
        """Use this method to verify if the created `ListElementsCompareType` is
        valid to use with PlotjugglerPublisher.

        Args:
            raise_exception_if_wrong (bool, optional): If set to True, will raise an exception if
                something is wrong, otherwise only prints to console. Defaults to True.

        Returns:
            bool: True, if data is of valid format for publishing using PlotjugglerPublisher.
        """
        try:
            assert isinstance(self.elements, list), "Field `elements` should be a list of strings."

            for e in self.elements:
                assert isinstance(
                    e, str
                ), f"Field `elements` should be a list of strings. Got type {type(e)} in list."

            assert self.values is not None, "Field `values` is not populated."

            n = len(self.elements)
            c_n = None
            if self.row_names is not None:
                assert isinstance(
                    self.row_names, list
                ), "Field `row_names` should be a list of strings."

                for c in self.row_names:
                    assert isinstance(
                        c, str
                    ), f"Field `row_names` should be a list of strings. Got type {type(c)} in list."
                c_n = len(self.row_names)
            type_error_msg = (
                "The `values` field should be a list of lists (of N floats), "
                "or a numpy array of shape [-1, N] where N = len(elements)."
            )
            if isinstance(self.values, list):
                if not isinstance(self.values[0], list):
                    raise TypeError(
                        type_error_msg
                        + f" Expected list of list. Received list of {type(self.values[0])}."
                    )
                for val in self.values:
                    assert len(val) == n, (
                        type_error_msg
                        + f" Expected list of list of shape [any, {n}]; Got [{len(self.values)},{len(val)}]. Culprit: {val}."
                    )

                if c_n is not None and len(self.values) != c_n:
                    raise TypeError(
                        type_error_msg
                        + f" When `row_names` is defined, expected list of list of list of shape [{len(self.values)}, {n}]; Got [{len(self.values)},{n}].",
                    )

            if isinstance(self.values, np.ndarray):
                if len(self.values.shape) != 2 or self.values.shape[1] != n:
                    raise TypeError(
                        type_error_msg + f" Expected shape: [any,{n}]; Got {self.values.shape}."
                    )
                if c_n is not None:
                    assert self.values.shape[0] == c_n, (
                        type_error_msg + f" Expected shape: [{c_n},{n}]; Got {self.values.shape}."
                    )
        except (TypeError, AssertionError) as e:
            if raise_exception_if_wrong:
                raise
            else:
                print(repr(e))
                return False

        return True


@dataclass
class JointStatesCompareType:
    """A custom data type defined to make comparison of values in JointStates datatypes easier.
    Use this type to publish data to plotjuggler using PlotjugglerPublisher, which will handle
    encoding of this data type to make it easier to compare elements in plotjuggler.
    This datatype will make it easier to compare joint positions, velocities and efforts of
    each joint from different sources.
    """

    joint_states_list: List[JointStates]
    row_names: List[str] = None

    def get_row_names(self) -> List[str]:
        if self.row_names is not None and len(self.row_names) == len(self.joint_states_list):
            return self.row_names
        else:
            return [f"data_{i}" for i in range(len(self.joint_states_list))]


@dataclass
class Pose3DCompareType:
    """A custom data type defined to make comparison of Pose3D datatypes easier.
    Use this type to publish data to plotjuggler using PlotjugglerPublisher, which will handle
    encoding of this data type to make it easier to compare elements in plotjuggler.
    This datatype will make it easier to compare x, y, z and roll, pitch, yaw values from
    different sources.
    """

    pose_list: List[Pose3D]
    row_names: List[str] = None

    def get_row_names(self) -> List[str]:
        if self.row_names is not None and len(self.row_names) == len(self.pose_list):
            return self.row_names
        else:
            return [f"data_{i}" for i in range(len(self.pose_list))]
