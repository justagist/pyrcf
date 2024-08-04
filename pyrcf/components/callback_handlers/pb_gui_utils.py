from typing import Tuple, Callable, TypeAlias, List
import pybullet as pb
import numpy as np

from ...utils.time_utils import RateTrigger, PythonPerfClock
from ...utils.math_utils import quat2rot

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


class PybulletGUIButton:
    """Create a button in Pybullet GUI, and check if it was pressed.
    Meant to be used in a loop."""

    def __init__(self, name: str, cid: int):
        """Create a button in Pybullet GUI, and check if it was pressed.
        Meant to be used in a loop.

            Args:
                name (str): Name on button.
                cid (int): The client id to connect to the physics simulator.
        """
        self.cid = cid
        self._id = pb.addUserDebugParameter(name, 1, 0, 1, physicsClientId=self.cid)
        self._val = 1

    def reset(self):
        """Not really useful anymore (legacy)"""
        self._val = 1

    def was_pressed(self) -> bool:
        """Checks if this button was pressed since the last time this method was called."""
        new_val = pb.readUserDebugParameter(self._id, physicsClientId=self.cid)
        triggered = new_val > self._val
        self._val = new_val
        return triggered

    def remove(self):
        """Remove the button from the gui."""
        try:
            pb.removeUserDebugItem(self._id, physicsClientId=self.cid)
        except pb.error:
            pass


class PybulletGUISlider:
    """Create a slider gui in pybullet."""

    def __init__(
        self,
        name: str,
        lower_lim: float,
        upper_lim: float,
        default_val: float,
        cid: int,
        mapping_function: Callable[[float], float] = lambda x: x,
    ):
        """Create a slider gui in pybullet.

        Args:
            name (str): Name for slider.
            lower_lim (float): The min value for slider.
            upper_lim (float): The max value for slider.
            default_val (float): Default starting value for slider.
            cid (int): The client id to connect to the physics simulator.
            mapping_function (Callable[[float], float], optional): Mapping to do for the
                value read from the slider (e.g. convert radian to degrees)
        """
        self.cid = cid
        self._id = pb.addUserDebugParameter(
            name,
            lower_lim,
            upper_lim,
            default_val,
            physicsClientId=self.cid,
        )
        assert callable(mapping_function)
        self._mapping_function = mapping_function
        self._last_recorded_val = None

    def get_value(self) -> float:
        """Read the value from the slider gui."""
        self._last_recorded_val = self._mapping_function(
            pb.readUserDebugParameter(self._id, physicsClientId=self.cid)
        )
        return self._last_recorded_val

    @property
    def last_read_value(self) -> float:
        return self._last_recorded_val

    def value_changed(self) -> bool:
        latest_val = self._last_recorded_val
        new_val = self.get_value()
        return new_val != latest_val

    def remove(self):
        """Remove the slider from gui."""
        try:
            pb.removeUserDebugItem(self._id, physicsClientId=self.cid)
        except pb.error:
            pass

    def set_value(self, value: float):
        raise NotImplementedError(
            "Pybullet slider values cannot be updated! Use TkInter sliders instead."
        )


class PybulletDebugFrameViz:
    """Draw a coordinate frame in pybullet as specified pose in the world."""

    def __init__(
        self,
        cid: int,
        position: Vector3D = None,
        orientation: QuatType = np.array([0, 0, 0, 1]),
        line_length: float = 0.2,
        line_width: float = 2,
        duration: float = 0.0,
        draw_at_init: bool = True,
    ):
        """Draw a coordinate frame in pybullet as specified pose in the world.

        Args:
            position (Vector3D): Position of coordinate frame in world.
            orientation (QuatType): Orientation quaternion of the frame in the world.
            cid (int): The physics server id to connect to.
            line_length (float, optional): Length of the lines in the world. Defaults to 0.2.
            line_width (float, optional): Line thickness. Defaults to 2.
            duration (float, optional): Lifetime for the frame. Defaults to 0.0 (infinite).
        """
        self._len = line_length
        self._lw = line_width
        self._duration = duration
        self._cid = cid

        self._line_ids = [-1, -1, -1]
        if draw_at_init:
            assert (
                position is not None and orientation is not None
            ), "Position and orientation for coordinate frame not provided."
            self._draw_lines(pos=position, ori=orientation)

    def _draw_lines(
        self,
        pos: Vector3D,
        ori: QuatType,
    ):
        rot_mat = quat2rot(ori).T
        toX = pos + self._len * rot_mat[0, :]
        toY = pos + self._len * rot_mat[1, :]
        toZ = pos + self._len * rot_mat[2, :]
        line_ids = self._line_ids
        self._line_ids[0] = pb.addUserDebugLine(
            pos,
            toX,
            [1, 0, 0],
            self._lw,
            self._duration,
            physicsClientId=self._cid,
            replaceItemUniqueId=line_ids[0],
        )
        self._line_ids[1] = pb.addUserDebugLine(
            pos,
            toY,
            [0, 1, 0],
            self._lw,
            self._duration,
            physicsClientId=self._cid,
            replaceItemUniqueId=line_ids[1],
        )
        self._line_ids[2] = pb.addUserDebugLine(
            pos,
            toZ,
            [0, 0, 1],
            self._lw,
            self._duration,
            physicsClientId=self._cid,
            replaceItemUniqueId=line_ids[2],
        )

    def update_frame_pose(self, position: Vector3D, orientation: QuatType):
        """Update the pose of an existing frame viz in pybullet.

        Args:
            position (Vector3D): Position of coordinate frame in world.
            orientation (QuatType): Orientation quaternion of the frame in the world.
        """
        self._draw_lines(pos=position, ori=orientation)

    def remove(self):
        """Remove the frame viz from gui."""

        try:
            for l_id in self._line_ids:
                pb.removeUserDebugItem(l_id, physicsClientId=self._cid)
        except pb.error:
            pass
        self._line_ids = [-1, -1, -1]


class PybulletText:
    """Add text to pybullet sim world gui."""

    def __init__(
        self,
        text_position: Tuple[float, float, float],
        default_text: str = "",
        text_color: Tuple[float, float, float] = (0, 0, 0),
        text_size: float = 3.0,
        lifetime: float = 0.0,
        cid: int = 0,
    ):
        """Add text to pybullet sim world gui.

        Args:
            text_position (Tuple[float, float, float]): 3D position to place the text in world
                coordinates.
            default_text (str, optional): Text to show. Defaults to "".
            text_color (Tuple[float, float, float], optional): RGB color code (range [0,1]).
                Defaults to (0, 0, 0).
            text_size (float, optional): Default text size. Defaults to 3.0.
            lifetime (float, optional): How long (in sec) the text should show. Defaults to 0.0
                (persistent).
            cid (int, optional): The client id to connect to the physics simulator. Defaults to 0.
        """
        self.cid = cid
        self._pos = text_position
        self._color = text_color
        self._size = text_size
        self._t = lifetime
        self._id = pb.addUserDebugText(
            default_text,
            self._pos,
            self._color,
            self._size,
            self._t,
            physicsClientId=self.cid,
        )

    def update_text(self, text: str):
        """Update the text string for this debug object.

        Args:
            text (str): The new text to replace with.
        """
        prev_id = self._id
        self._id = pb.addUserDebugText(
            str(text),
            self._pos,
            self._color,
            self._size,
            self._t,
            replaceItemUniqueId=prev_id,
            physicsClientId=self.cid,
        )

    def remove(self):
        """Remove text from pybullet."""
        try:
            pb.removeUserDebugItem(self._id, physicsClientId=self.cid)
        except pb.error:
            pass


class PybulletTextWithRateTrigger(PybulletText):
    """Add text to pybullet sim world gui, that will only change at the
    specified rate (even if `update_text` is called at a higher frequency).
    This is useful if this method is called in a high frequency loop such as
    a control loop, and text has to be rendered only at a lower frequency.
    """

    def __init__(
        self,
        text_position: Tuple[float, float, float],
        rate: float = 1,
        default_text: str = "",
        text_color: Tuple[float, float, float] = (0, 0, 0),
        text_size: float = 3.0,
        lifetime: float = 0.0,
        cid: int = 0,
    ):
        """Add text to pybullet sim world gui, that will only change at the
        specified rate (even if `update_text` is called at a higher frequency).
        This is useful if this method is called in a high frequency loop such as
        a control loop, and text has to be rendered only at a lower frequency.

        Args:
            text_position (Tuple[float, float, float]): 3D position to place the text in world
                coordinates.
            rate (float, optional): Update rate for text updating. Defaults to 1.
            default_text (str, optional): Text to show. Defaults to "".
            text_color (Tuple[float, float, float], optional): RGB color code (range [0,1]).
                Defaults to (0, 0, 0).
            text_size (float, optional): Default text size. Defaults to 3.0.
            lifetime (float, optional): How long (in sec) the text should show. Defaults to 0.0
                (persistent).
            cid (int, optional): The client id to connect to the physics simulator. Defaults to 0.
        """
        self._trigger = RateTrigger(rate=rate, clock=PythonPerfClock())
        super().__init__(
            text_position=text_position,
            default_text=default_text,
            text_color=text_color,
            text_size=text_size,
            lifetime=lifetime,
            cid=cid,
        )

    def update_text(self, text: str):
        """Update the text string for this debug object. Will only be triggered if the specified
        rate is satisified (lower bound).

        Args:
            text (str): The new text to replace with.
        """
        if self._trigger():
            return super().update_text(text)


class PybulletDebugPoints:
    """Add 3D points to pybullet sim world gui."""

    def __init__(
        self,
        point_positions: List[Vector3D] = None,
        point_size: float = 1.0,
        rgb: Tuple[float, float, float] = (1, 0, 0),
        lifetime: float = 0.0,
        cid: int = 0,
    ):
        """Add 3D points to pybullet sim world gui.

        Args:
            point_positions (List[Vector3D], optional): Initial positions of the points.
                Defaults to None.
            point_size (float, optional): _description_. Defaults to 1.0.
            rgb (Tuple[float, float, float], optional): RGB color code for the points.
                Defaults to (1, 0, 0).
            lifetime (float, optional): How long (in sec) the points should show. Defaults to 0.0
                (persistent).
            cid (int, optional): The client id to connect to the physics simulator. Defaults to 0.
        """

        self._lifetime = lifetime
        self._size = point_size
        self._rgb = rgb
        self._cid = cid

        self._points_id = -1
        if point_positions is not None:
            self._draw_points(point_positions=point_positions)

    def _draw_points(self, point_positions: List[Vector3D]):

        self._points_id = pb.addUserDebugPoints(
            pointPositions=point_positions,
            pointColorsRGB=[self._rgb] * len(point_positions),
            pointSize=self._size,
            lifeTime=self._lifetime,
            replaceItemUniqueId=self._points_id,
            physicsClientId=self._cid,
        )

    def update_point_positions(self, point_positions: List[Vector3D]):
        """Update the positions of the debug points in pybullet.

        Args:
            point_positions (List[Vector3D]): List of positions for the points in world.
        """
        self._draw_points(point_positions=point_positions)

    def remove(self):
        """Remove the debug points object from pybullet."""
        try:
            pb.removeUserDebugItem(self._points_id, physicsClientId=self._cid)
        except pb.error:
            pass
