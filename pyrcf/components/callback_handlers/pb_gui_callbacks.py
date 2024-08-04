## CALLBACK FUNCTIONALITY UTILS WITH PbGUIs
from typing import Tuple, List, Callable, TypeAlias
import numpy as np

from .base_callbacks import CustomCallbackBase
from .pb_gui_utils import (
    PybulletGUISlider,
    PybulletGUIButton,
    PybulletDebugFrameViz,
    PybulletDebugPoints,
)

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


class PbGUICallback(CustomCallbackBase):
    """Abstraction for all GUICallbacks defined using Pybullet GUI."""


class PbGUISliderCallback(PbGUICallback):
    """Create a slider in pybullet and specify the callback to apply the slider value to."""

    def __init__(
        self,
        slider_name: str,
        slider_lower_lim: float,
        slider_upper_lim: float,
        slider_default_val: float,
        callback: Callable[[float], None],
        cid: int,
        run_only_if_value_changed: bool = True,
        mapping_function: Callable[[float], float] = lambda x: x,
    ):
        """Create a slider in pybullet and specify the callback to apply the slider value to.

        Args:
            slider_name (str): Name for slider.
            slider_lower_lim (float): The min value for slider.
            slider_upper_lim (float): The max value for slider.
            slider_default_val (float): Default starting value for slider.
            callback (Callable[[float], None]): Handle to a function that takes as input
                the slider value (no other arguments).
            cid (int): The client id to connect to the physics simulator.
            run_only_if_value_changed (bool, optional): The callback will be executed only if
                slider value has changed since the last time it was checked (i.e. `run_once`
                was called for this object).
            mapping_function (Callable[[float], float], optional): Mapping to do for the
                value read from the slider (e.g. convert radian to degrees)
        """
        self._slider = PybulletGUISlider(
            name=slider_name,
            lower_lim=slider_lower_lim,
            upper_lim=slider_upper_lim,
            default_val=slider_default_val,
            cid=cid,
            mapping_function=mapping_function,
        )
        self._check_first = run_only_if_value_changed
        self._callback = callback

    def run_once(self) -> None:
        """This method has to be called for the callback to be executed."""
        if self._check_first and not self._slider.value_changed():
            return
        self._callback(self._slider.get_value())

    def cleanup(self):
        self._slider.remove()


class PbMultiGUISliderSingleCallback(PbGUICallback):
    """Create multiple sliders in pybullet and use values from all the sliders in a single
    callback."""

    def __init__(
        self,
        slider_names: List[str],
        slider_lower_lims: List[float],
        slider_upper_lims: List[float],
        slider_default_vals: List[float],
        callback: Callable[[List[float]], None],
        cid: int,
        run_only_if_value_changed: bool = True,
        mapping_functions: Callable[[float], float] | List[Callable[[float], float]] = lambda x: x,
    ):
        """Create multiple sliders and use values from all the sliders in a single callback.

        Args:
            slider_names (List[str]): Names for slider.
            slider_lower_lims (List[float]): Corresponding min values for sliders.
            slider_upper_lims (List[float]): Corresponding max values for sliders.
            slider_default_vals (List[float]): Corresponding default values for sliders.
            callback (Callable[[List[float]], None]): Handle to a function that takes as input
                the slider values (list of floats).
            cid (int): The client id to connect to the physics simulator.
            run_only_if_value_changed (bool, optional): The callback will be executed only if
                slider value has changed since the last time it was checked (i.e. `run_once`
                was called for this object).
            mapping_functions (Callable[[float], float] | List[Callable[[float], float]], optional):
                Mapping to do for the value read from the slider (e.g. convert radian to degrees).
        """
        self._sliders: List[PybulletGUISlider] = []
        if callable(mapping_functions):
            mapping_functions = [mapping_functions] * len(slider_names)
        for n, name in enumerate(slider_names):
            self._sliders.append(
                PybulletGUISlider(
                    name=name,
                    lower_lim=slider_lower_lims[n],
                    upper_lim=slider_upper_lims[n],
                    default_val=slider_default_vals[n],
                    cid=cid,
                    mapping_function=mapping_functions[n],
                )
            )
        self._check_first = run_only_if_value_changed
        self._callback = callback

    def run_once(self):
        """This method has to be called for the callback to be executed."""
        if self._check_first:
            for s in self._sliders:
                if s.value_changed():
                    break
            else:
                return
        vals = [slider.get_value() for slider in self._sliders]
        self._callback(vals)

    def cleanup(self):
        for slider in self._sliders:
            slider.remove()
        self._sliders.clear()

    def get_sliders(self) -> List[PybulletGUISlider]:
        """Get a list of all slider objects.

        Returns:
            List[PybulletGUISlider]: List of slider objects.
        """
        return self._sliders


class PbGUIButtonCallback(PbGUICallback):
    """Create a button in pybullet and specify the callback to apply the slider value to."""

    def __init__(self, button_name: str, callback: Callable[[], None], cid: int):
        """Create a button in pybullet and specify the callback to apply the slider value to.

        Args:
            button_name (str): Name of button.
            callback (Callable[[], None]): Handle to a function that should be executed when
                the button is pressed.
            cid (int): The client id to connect to the physics simulator.
        """
        self._button = PybulletGUIButton(name=button_name, cid=cid)
        self._callback = callback

    def run_once(self):
        """This method has to be called for checking button press and for the
        callback to be executed."""
        if self._button.was_pressed():
            self._callback()

    def cleanup(self):
        self._button.remove()


class PbDebugFrameVizCallback(PbGUICallback):
    """Draw a coordinate frame in pybullet that changes its pose based on the pose
    returned by a callback function."""

    def __init__(
        self,
        callback: Callable[[], Tuple[Vector3D, QuatType]],
        cid: int,
        line_length: float = 0.2,
        line_width: float = 2,
        duration: float = 0.0,
    ):
        """Draw a coordinate frame in pybullet that changes its pose based on the pose
        returned by a callback function.

        Args:
            callback (Callable[[Vector3D, QuatType], None]): Handle to a function that returns
                a position and orientation target for the coordinate frame to visualize.
            cid (int): The physics server id to connect to.
            line_length (float, optional): Length of the lines in the world. Defaults to 0.2.
            line_width (float, optional): Line thickness. Defaults to 2.
            duration (float, optional): Lifetime for the frame. Defaults to 0.0 (infinite).
        """
        self._frame_viz = PybulletDebugFrameViz(
            cid=cid,
            position=None,
            orientation=None,
            line_width=line_width,
            line_length=line_length,
            duration=duration,
            draw_at_init=False,
        )
        assert callable(callback)
        self._callback = callback

    def run_once(self):
        """This method has to be called for for getting the pose from the callback
        function and updating the frame visualiser's pose."""
        p, q = self._callback()
        self._frame_viz.update_frame_pose(position=p, orientation=q)

    def cleanup(self):
        self._frame_viz.remove()


class PbDebugPointsCallback(PbGUICallback):
    """Visualise point(s)/sphere(s) in pybullet exposing a callback to change its position
    in the world.
    """

    def __init__(
        self,
        callback: Callable[[], List[Vector3D]],
        cid: int,
        point_positions: List[Vector3D] = None,
        point_size: float = 1.0,
        rgb: Tuple[float, float, float] = (1, 0, 0),
        lifetime: float = 0.0,
    ):
        """Visualise point(s)/sphere(s) in pybullet exposing a callback to change its position
        in the world.

        Args:
            callback (Callable[[], List[Vector3D]]): Function returning positions of the point.
            cid (int): The physics server id to connect to.
            point_positions (List[Vector3D], optional): Starting positions for the points. Defaults
                to None.
            point_size (float, optional): Size of points. Defaults to 1.0.
            rgb (Tuple[float, float, float], optional): RGB color code. Defaults to (1, 0, 0).
            lifetime (float, optional): Lifetime for the points in seconds. Defaults to 0.0
                (persistent).
        """
        self._points_viz = PybulletDebugPoints(
            point_positions=point_positions,
            point_size=point_size,
            rgb=rgb,
            lifetime=lifetime,
            cid=cid,
        )
        assert callable(callback)
        self._callback = callback

    def run_once(self):
        self._points_viz.update_point_positions(point_positions=self._callback())

    def cleanup(self):
        self._points_viz.remove()
