from dataclasses import dataclass, field
from typing import Tuple, Mapping, TypeAlias, List
import copy
import numpy as np

from .tf_types import QuatType, Vector3D, Twist, Pose3D

QuatTrajType: TypeAlias = np.ndarray
"""Numpy array representating a trajectory of quaternions in format [[x,y,z,w]].
Shape: [t, 4], where t is the number of trajectory points.
"""

Vector3DTrajType: TypeAlias = np.ndarray
"""Numpy array representating a trajectory of 3D vectors in format [[x,y,z]].
Shape: [t, 3], where t is the number of trajectory points.
"""


@dataclass
class PointMotion:
    """Container to hold position, velocity, acceleration data of a point moving in 3D space."""

    position: np.ndarray = None
    """3D position of point."""
    velocity: np.ndarray = None
    """3D velocity of point."""
    acceleration: np.ndarray = None
    """3D acceleration of point."""

    def __post_init__(self):
        self.__do_checks()

    def __do_checks(self):
        if self.position is not None:
            assert (
                len(self.position) == 3
            ), f"Position dimension should be 3 for object of type {self.__class__.__name__}."
        if self.velocity is not None:
            assert (
                len(self.velocity) == 3
            ), f"Velocity dimension should be 3 for object of type {self.__class__.__name__}."
        if self.acceleration is not None:
            assert (
                len(self.acceleration) == 3
            ), f"Acceleration dimension should be 3 for object of type {self.__class__.__name__}."

    def __eq__(self, other: "PointMotion") -> bool:
        if self.position is not None and not np.all(self.position == other.position):
            return False
        if self.velocity is not None and not np.all(self.velocity == other.velocity):
            return False
        if self.acceleration is not None and np.all(self.acceleration == self.acceleration):
            return False
        return True

    def __setattr__(self, prop, val):
        if prop not in ["position", "velocity", "acceleration"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {prop} in object ot type {self.__class__.__name__}."
            )
        if val is not None:
            val = np.array(val)
        super().__setattr__(prop, val)
        self.__do_checks()


@dataclass
class FrameMotion:
    """Container to hold pose, vel, acc data for an object/frame/vector moving in 3D space."""

    pose: Pose3D = None
    velocity: Twist = None
    acceleration: Twist = None

    def __eq__(self, other: "FrameMotion") -> bool:
        return (
            self.pose == other.pose
            and self.velocity == other.velocity
            and self.acceleration == other.acceleration
        )

    def __setattr__(self, prop, val):
        if prop not in ["pose", "velocity", "acceleration"]:
            raise NameError(
                f"Tried to assign value to illegal parameter {prop} in object ot type {self.__class__.__name__}."
            )
        if val is not None:
            if prop == "pose":
                assert isinstance(val, Pose3D)
            else:
                assert isinstance(val, Twist)
        super().__setattr__(prop, val)


def _get_closest_timestamp_idx(
    timestamps: np.ndarray,
    t: float,
    traj_start_time: float,
    traj_end_time: float,
    only_if_time_passed: bool,
    respect_traj_start_time: bool,
    respect_traj_end_time: bool,
) -> int:
    """NOTE: for internal use only"""
    if respect_traj_start_time and t < traj_start_time:
        raise ValueError(f"Trajectory does not have any value before timestamp={traj_start_time}.")
    if respect_traj_end_time and t > traj_end_time:
        raise ValueError(f"Trajectory does not have any value after timestamp={traj_end_time}.")
    # IF your array is sorted and is very large, this is a much faster search than np.where
    idx = np.searchsorted(timestamps, t, side="left")
    if idx > 0 and (
        idx == len(timestamps)
        # the check t < timestamps[idx] may have floating point precision errors
        # e.g., 1.99999999999999999999 < 2 may return False, because they could be equal
        or (only_if_time_passed and t < timestamps[idx])
        or np.fabs(t - timestamps[idx - 1]) <= np.fabs(t - timestamps[idx])
    ):
        return idx - 1
    else:
        return idx


@dataclass
class GeneralisedTrajectory:
    """Container for holding generalised N-D trajectories."""

    traj: np.ndarray
    """Sequence of ND positions in the trajectory. Shape [t,N]."""
    timestamps: np.ndarray = None
    """Optionally, each point in the trajectory can be timestamped with this [t] dim vector."""

    dim_names: List[str] = None
    """The name of each dimension of the N-D trajectory (e.g. joint names if this object contains
    joint trajectory)."""

    traj_start_time: float = field(default=None, init=False)
    """Start time for this trajectory. Only available if initialised with `timestamps`."""
    traj_end_time: float = field(default=None, init=False)
    """End time of this trajectory. Only available if initialised with `timestamps`."""

    def __post_init__(self):
        if self.timestamps is not None:
            self.traj_start_time = self.timestamps[0]
            self.traj_end_time = self.timestamps[-1]

    def __getitem__(self, idx: int) -> Tuple[float, np.ndarray]:
        """Subscripting this object will return the time, trajectory point at the specified index.

        Args:
            idx (int): The integer index.

        Returns:
            Tuple[float, np.ndarray]: Timestamp, trajectory point at the specified index.
        """
        assert (
            idx < self.traj.shape[0]
        ), f"Index {idx} out of bounds for trajectory of size {self.traj.shape[0]}."
        t = None if self.timestamps is None else self.timestamps[idx]
        return (t, self.traj[idx, :])

    def get_point_at_time(self, t: float) -> Tuple[int, float, np.ndarray]:
        """Get the index in list, timestamp and trajectory point from this trajectory given a
        valid timestamp.

        Args:
            t (float): The exact timestamp to retrieve the pose for.

        Raises:
            ValueError: Raised when the exact match for t is not found in this trajectory.

        Returns:
            Tuple[int, float, np.ndarray]: Index in list, timestamp, trajectory point at the specified time.
                NOTE: This method returns the redundant timestamp object as the first value in the
                returned tuple. This is to keep consistency with the `get_point_at_approx_time()` method.
                The value of timestamp will be the same as the input `t`, and can be ignored.
        """
        assert (
            self.timestamps is not None
        ), "This GeneralisedTrajectory object does not have `timestamps` attribute defined."
        try:
            idx = np.where(self.timestamps == t)[0][0]
        except IndexError as exc:
            raise ValueError(
                f"No exact match for timestamp {t} in trajectory. Try using `get_point_at_approx_time(t={t})`."
            ) from exc
        return idx, *self.__getitem__(idx)

    def get_point_at_approx_time(
        self,
        t: float,
        only_if_time_passed: bool = True,
        respect_traj_start_time: bool = True,
        respect_traj_end_time: bool = False,
    ) -> Tuple[int, float, np.ndarray]:
        """Get the index in list, timestamp and trajectory point at a time closest to the specified t.

        Args:
            t (float): The desired timestamp.
            only_if_time_passed (bool, optional): When this is enabled, the returned values are
                ensured to be before the next timestamp in the trajectory. This is useful
                when querying trajectory points in a loop with incrementing timestamp, and you
                want to be sure that the returned values are not from the future. This way a
                trajectory value is returned only if the specified time is >= its corresponding
                timestamp and before the next timestamp. E.g. this way if a planner/controller,
                uses this method to query a trajectory at a time, it does not return a value from
                future. Defaults to True.
            respect_traj_start_time (bool, optional): If set to true, raises ValueError if queried
                time is before self.traj_start_time. Defaults to True.
            respect_traj_end_time (bool, optional): If set to true, raises ValueError if queried
                time is after self.traj_end_time. Defaults to False.

        Returns:
            Tuple[int, float, np.ndarray]: Index in list, timestamp, trajectory point from this object.
        """

        assert (
            self.timestamps is not None
        ), "This GeneralisedTrajectory does not have `timestamps` attribute defined."
        idx = _get_closest_timestamp_idx(
            timestamps=self.timestamps,
            t=t,
            traj_start_time=self.traj_start_time,
            traj_end_time=self.traj_end_time,
            only_if_time_passed=only_if_time_passed,
            respect_traj_start_time=respect_traj_start_time,
            respect_traj_end_time=respect_traj_end_time,
        )
        return idx, *self.__getitem__(idx)


@dataclass
class Trajectory3D:
    """Container for holding 3D pose trajectories.

    NOTE: When creating this object, use `validate_trajectory=True` to assert trajectory validity.
        Defaults to True. Disable this to speed up construction.
    """

    position_traj: Vector3DTrajType
    """Sequence of 3D positions in the trajectory. Shape [t,3]."""
    orientation_traj: QuatTrajType
    """Sequence of quaternion orientations in the trajectory. Shape [t,4]."""
    timestamps: np.ndarray = None
    """Optionally, each point in the trajectory can be timestamped with this [t] dim vector."""

    frame_name: str = ""
    """The name of the frame whose trajectory is defined in this object."""

    validate_trajectory: bool = field(default=True, init=True, repr=False)
    """If this is enabled, trajectory validity will be checked after construction. Disable this for
    quicker initialisations. Defaults to True."""

    traj_start_time: float = field(default=None, init=False)
    """Start time for this trajectory. Only available if initialised with `timestamps`."""
    traj_end_time: float = field(default=None, init=False)
    """End time of this trajectory. Only available if initialised with `timestamps`."""

    def __post_init__(self):
        if self.validate_trajectory:
            # check if position traj and orientation traj lengths match
            assert (
                self.position_traj.shape[0] == self.orientation_traj.shape[0]
            ), f"Invalid trajectory for frame {self.frame_name}.."

            if self.timestamps is not None:
                # check if timestamps length makes sense
                assert (
                    self.timestamps.shape[0] == self.position_traj.shape[0]
                ), f"Timestamps array size ({self.timestamps.shape[0]}) does not match trajectory size ({self.position_traj.shape[0]}) for frame {self.frame_name}."
                # check if timestamps increase monotonically
                assert np.all(
                    self.timestamps == np.sort(self.timestamps)
                ), "Timestamps should be strictly increasing."

            # check position trajectory dimension
            assert (
                len(self.position_traj.shape) == 2 and self.position_traj.shape[1] == 3
            ), f"Position trajectory should be of shape [t,3] for frame {self.frame_name}."

            # check quaternion trajectory dimension
            assert (
                len(self.orientation_traj.shape) == 2 and self.orientation_traj.shape[1] == 4
            ), f"Orientation quaternions trajectory should be of shape [t,4] for frame {self.frame_name}."

            # check quaternion validity for each quaternion
            for nq in range(self.orientation_traj.shape[0]):
                assert np.isclose(
                    np.linalg.norm(self.orientation_traj[nq, :]), 1.0
                ), f"Quaternion at index {nq} in trajectory is not normalised for frame {self.frame_name}. Culprit: {self.orientation_traj[nq,:]}."
        if self.timestamps is not None:
            self.traj_start_time = self.timestamps[0]
            self.traj_end_time = self.timestamps[-1]

    def __getitem__(self, idx: int) -> Tuple[float, Vector3D, QuatType]:
        """Subscripting this object will return the time, position, orientation values at the specified index.

        Args:
            idx (int): The integer index.

        Returns:
            Tuple[float, Vector3D, QuatType]: Timestamp, position, orientation at the specified index.
        """
        assert (
            idx < self.position_traj.shape[0]
        ), f"Index {idx} out of bounds for trajectory of size {self.position_traj.shape[0]}."
        t = None if self.timestamps is None else self.timestamps[idx]
        return (
            t,
            self.position_traj[idx, :],
            self.orientation_traj[idx, :],
        )

    def get_pose_at_time(self, t: float) -> Tuple[int, float, Vector3D, QuatType]:
        """Get the index in list, timestampe, position and orientation values from this trajectory
        given a valid timestamp.

        Args:
            t (float): The exact timestamp to retrieve the pose for.

        Raises:
            ValueError: Raised when the exact match for t is not found in this trajectory.

        Returns:
            Tuple[int, float, Vector3D, QuatType]: Index in list, timestamp, position and quaternion
                values at the specified time.
                NOTE: This method returns the redundant timestamp object as the first value in the
                returned tuple. This is to keep consistency with the `get_pose_at_approx_time()`
                method. The value of timestamp will be the same as the input `t`, and can be
                ignored.
        """
        assert (
            self.timestamps is not None
        ), "This Trajectory3D does not have `timestamps` attribute defined."
        try:
            idx = np.where(self.timestamps == t)[0][0]
        except IndexError as exc:
            raise ValueError(
                f"No exact match for timestamp {t} in trajectory. Try using `get_pose_at_approx_time(t={t})`."
            ) from exc
        return idx, *self.__getitem__(idx)

    def get_pose_at_approx_time(
        self,
        t: float,
        only_if_time_passed: bool = True,
        respect_traj_start_time: bool = True,
        respect_traj_end_time: bool = False,
    ) -> Tuple[int, float, Vector3D, QuatType]:
        """Get the index, timestamp, position and orientation value at a time closest to the
        specified t.

        Args:
            t (float): The desired timestamp.
            only_if_time_passed (bool, optional): When this is enabled, the returned values are
                ensured to be before the next timestamp in the trajectory. This is useful
                when querying trajectory points in a loop with incrementing timestamp, and you
                want to be sure that the returned values are not from the future. This way a
                pose value is returned only if the specified time is >= its corresponding
                timestamp and before the next timestamp. E.g. this way if a planner/controller,
                uses this method to query a pose at a time, it does not return a value from the
                future. Defaults to True.
            respect_traj_start_time (bool, optional): If set to true, raises ValueError if queried
                time is before self.traj_start_time. Defaults to True.
            respect_traj_end_time (bool, optional): If set to true, raises ValueError if queried
                time is after self.traj_end_time. Defaults to False.

        Returns:
            Tuple[int, float, Vector3D, QuatType]: Index in list, closest timestamp, position,
                and quaternion from this Trajectory3D object.
        """

        assert (
            self.timestamps is not None
        ), "This Trajectory3D does not have `timestamps` attribute defined."
        idx = _get_closest_timestamp_idx(
            timestamps=self.timestamps,
            t=t,
            traj_start_time=self.traj_start_time,
            traj_end_time=self.traj_end_time,
            only_if_time_passed=only_if_time_passed,
            respect_traj_start_time=respect_traj_start_time,
            respect_traj_end_time=respect_traj_end_time,
        )
        return idx, *self.__getitem__(idx)


@dataclass
class MultiFrameTrajectory3D:
    """Container for holding synchronised trajectories for multiple frames.

    This object is also a single-use iterator, and can therefore be used in a loop, or using
    the `next(mf_traj)` to get the next `__getitem__` call. However, being a single-use
    iterator, it cannot be used again as an iterable once it hits `StopIteration`.

    WARNING
    -------
    Avoid construction in control loop. Can be heavy.

    Args:
        trajectories (List[Trajectory3D]): List of Trajectory 3D object holding pose trajectories
            for different frames.
        timestamps (np.ndarray): Timestamps associated with the waypoints in the trajectories.
            Each point in the trajectory (for all frames) is timestamped with this [t] dim vector.
            NOTE: all trajectories will be assumed to be using the same timestamps defined here.
    """

    trajectories: List[Trajectory3D]
    """List of Trajectory 3D object holding pose trajectories for different frames."""

    timestamps: np.ndarray
    """Each point in the trajectory (for all frames) is timestamped with this [t] dim vector.
    NOTE: all trajectories will be assumed to be using the same timestamps defined here.
    """

    traj_start_time: float = field(default=None, init=False)
    """Start time for this trajectory."""
    traj_end_time: float = field(default=None, init=False)
    """End time of this trajectory."""

    _curr_idx: int = field(repr=False, init=False)
    _max_idx: int = field(repr=False, init=False)

    @classmethod
    def createFromKeywordArgs(
        cls: "MultiFrameTrajectory3D",
        timestamps: List[float] | np.ndarray,
        **kwargs: Tuple[Vector3DTrajType, QuatTrajType],
    ) -> "MultiFrameTrajectory3D":
        """Classmethod to construct a MultiFrameTrajectory3D object using keywords for each frame'
        trajectory.

        Args:
            timestamps (List[float] | np.ndarray): List of timestamps to be used for all frames
            **kwargs (Tuple[Vector3DTrajType, QuatTrajType]): Each keyword argument should be the
                name of the frame mapping to their corresponding position and orientation
                trajectory. e.g.

        MultiFrameTrajectory3D.createFromKeywordArgs(timestamps = ts, ee_1 = [postraj_1, quattraj1],
            some_other_frame = [postraj_2, quattraj_2])

        Returns:
            MultiFrameTrajectory3D: A MultiFrameTrajectory3D object with the specified frame
            trajectories.
        """
        assert (
            len(kwargs) > 0
        ), "Trajectory for frames should be passed using the frame name as keyword."

        traj_list = []

        for frame_name, kwarg in kwargs.items():
            assert (
                len(kwarg) == 2
            ), "Each frame (key) should have as value a tuple: [list of positions, list of quaternions]."
            position_traj = kwarg[0]
            quat_traj = kwarg[1]

            traj = Trajectory3D(
                position_traj=position_traj,
                orientation_traj=quat_traj,
                timestamps=timestamps,
                frame_name=frame_name,
                validate_trajectory=True,
            )

            traj_list.append(copy.deepcopy(traj))

        return cls(trajectories=traj_list, timestamps=timestamps)

    def __post_init__(self):
        if not isinstance(self.timestamps, np.ndarray):
            self.timestamps = np.array(self.timestamps)

        assert len(self.timestamps.shape) == 1

        self.traj_start_time = self.timestamps[0]
        self.traj_end_time = self.timestamps[-1]

        # check trajectory length for each frame
        for traj in self.trajectories:
            assert (
                self.timestamps.shape[0]
                == traj.position_traj.shape[0]
                == traj.orientation_traj.shape[0]
            )
            # check if each trajectory has a valid frame name
            assert (
                traj.frame_name is not None and traj.frame_name != ""
            ), "Each trajectory should have a defined frame_name"

        self._curr_idx = -1
        self._max_idx = self.timestamps.shape[0]

    def __getitem__(self, idx: int) -> Tuple[float, Mapping[str, Tuple[Vector3D, QuatType]]]:
        """Subscripting this object will return the timestamp, as well pose per frame at the
        specified index.

        Args:
            idx (int): The integer index.

        Returns:
            Tuple[float, Mapping[str, Tuple[Vector3D, QuatType]]]: Timestamp, Dictionary with
                mapping from frame_name -> [position, quaternion].
        """
        assert (
            idx < self.timestamps.shape[0]
        ), f"Index {idx} out of bounds for trajectory of size {self.timestamps.shape[0]}."
        return (
            self.timestamps[idx],
            {
                traj.frame_name: [
                    traj.position_traj[idx, :],
                    traj.orientation_traj[idx, :],
                ]
                for traj in self.trajectories
            },  # TODO: would this be mutable?? # pylint: disable=W0511
        )

    def get_frame_poses_at_time(
        self, t: float
    ) -> Tuple[int, float, Mapping[str, Tuple[Vector3D, QuatType]]]:
        """Get the pose for each frame from this object given a valid timestamp.

        Args:
            t (float): The timestamp to retrieve the pose for.

        Raises:
            ValueError: Raised when the exact match for t is not found in this trajectory.

        Returns:
            Tuple[int, float, Mapping[str, Tuple[Vector3D, QuatType]]]: Index in list, Timestamp,
                Dictionary with mapping from frame_name -> [position, quaternion] at the specified
                timestamp.
                NOTE: This method returns the redundant timestamp object as the first value in the
                returned tuple. This is to keep consistency with the `get_pose_at_approx_time()`
                method. The value of timestamp will be the same as the input `t`, and can be
                ignored.
        """
        try:
            idx = np.where(self.timestamps == t)[0][0]
        except IndexError as exc:
            raise ValueError(
                f"No exact match for timestamp {t} in trajectory. Try using `get_pose_at_approx_time(t={t})`."
            ) from exc
        return idx, *self.__getitem__(idx)

    def get_frame_poses_at_approx_time(
        self,
        t: float,
        only_if_time_passed: bool = True,
        respect_traj_start_time: bool = True,
        respect_traj_end_time: bool = False,
    ) -> Tuple[int, float, Mapping[str, Tuple[Vector3D, QuatType]]]:
        """Get the pose for each frame at a time closest to the specified t.

        Args:
            t (float): The desired timestamp.
            only_if_time_passed (bool, optional): When this is enabled, the returned values are
                ensured to be before the next timestamp in the trajectory. This is useful
                when querying trajectory points in a loop with incrementing timestamp, and you
                want to be sure that the returned values are not from the future. This way a
                pose value is returned only if the specified time is >= its corresponding
                timestamp and before the next timestamp. E.g. this way if a planner/controller,
                uses this method to query a pose at a time, it does not return a value from the
                future. Defaults to True.
            respect_traj_start_time (bool, optional): If set to true, raises ValueError if queried
                time is before self.traj_start_time. Defaults to True.
            respect_traj_end_time (bool, optional): If set to true, raises ValueError if queried
                time is after self.traj_end_time. Defaults to False.

        Returns:
            Tuple[int, float, Mapping[str, Tuple[Vector3D, QuatType]]]: Index in list, Timestamp,
                Dictionary with mapping from frame_name -> [position, quaternion] that is
                closest to the specified time in this MultiFrameTrajectory3D object.
        """
        idx = _get_closest_timestamp_idx(
            timestamps=self.timestamps,
            t=t,
            traj_start_time=self.traj_start_time,
            traj_end_time=self.traj_end_time,
            only_if_time_passed=only_if_time_passed,
            respect_traj_start_time=respect_traj_start_time,
            respect_traj_end_time=respect_traj_end_time,
        )
        return idx, *self.__getitem__(idx)

    def __iter__(self):
        return self

    def __next__(self):
        self._curr_idx += 1
        if self._curr_idx >= self._max_idx:
            raise StopIteration
        return self.__getitem__(self._curr_idx)
