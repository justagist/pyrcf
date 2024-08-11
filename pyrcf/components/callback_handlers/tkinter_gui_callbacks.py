## CALLBACK FUNCTIONALITY UTILS WITH PbGUIs
from typing import Callable, List
from numbers import Number

from .base_callbacks import CustomCallbackBase
from ...utils.gui_utils.tkinter_gui_utils import (
    TkinterGUISlider,
    TkinterWidgetMaster,
    TkinterGUIButton,
)


class TkGUICallback(CustomCallbackBase):
    """Abstraction for all GUICallbacks defined using Tkinter widgets."""


class TkGUISliderCallback(TkGUICallback):
    """Create a slider in pybullet and specify the callback to apply the slider value to."""

    def __init__(
        self,
        slider_name: str,
        slider_lower_lim: float,
        slider_upper_lim: float,
        slider_default_val: float,
        callback: Callable[[float], None],
        tk_master: TkinterWidgetMaster = None,
        mapping_function: Callable[[float], float] = lambda x: x,
        slider_resolution: float = 0.1,
    ):
        """Create a slider in pybullet and specify the callback to apply the slider value to.

        Args:
            slider_name (str): Name for slider.
            slider_lower_lim (float): The min value for slider.
            slider_upper_lim (float): The max value for slider.
            slider_default_val (float): Default starting value for slider.
            callback (Callable[[float], None]): Handle to a function that takes as input
                the slider value (no other arguments).
            mapping_function (Callable[[float], float], optional): Mapping to do for the
                value read from the slider (e.g. convert radian to degrees)
        """
        self._slider = TkinterGUISlider(
            name=slider_name,
            lower_lim=slider_lower_lim,
            upper_lim=slider_upper_lim,
            default_val=slider_default_val,
            tk_master=tk_master,
            mapping_function=mapping_function,
            slider_resolution=slider_resolution,
        )
        self._callback = callback

    def run_once(self) -> None:
        """This method has to be called for the callback to be executed."""
        self._callback(self._slider.get_value())

    def remove_gui_object(self):
        self._slider.remove()


class TkMultiGUISliderSingleCallback(TkGUICallback):
    """Create multiple sliders and use values from all the sliders in a single callback."""

    def __init__(
        self,
        slider_names: List[str],
        slider_lower_lims: List[float],
        slider_upper_lims: List[float],
        slider_default_vals: List[float],
        callback: Callable[[List[float]], None],
        tk_master: TkinterWidgetMaster = None,
        mapping_functions: Callable[[float], float] | List[Callable[[float], float]] = lambda x: x,
        slider_resolutions: List[float] | float = 0.1,
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
            mapping_functions (Callable[[float], float] | List[Callable[[float], float]], optional):
                Mapping to do for the value read from the slider (e.g. convert radian to degrees).
        """
        if isinstance(slider_resolutions, Number):
            slider_resolutions = [slider_resolutions] * len(slider_names)
        self._sliders: List[TkinterGUISlider] = []
        if callable(mapping_functions):
            mapping_functions = [mapping_functions] * len(slider_names)
        for n, name in enumerate(slider_names):
            self._sliders.append(
                TkinterGUISlider(
                    name=name,
                    lower_lim=slider_lower_lims[n],
                    upper_lim=slider_upper_lims[n],
                    default_val=slider_default_vals[n],
                    tk_master=tk_master,
                    mapping_function=mapping_functions[n],
                    slider_resolution=slider_resolutions[n],
                )
            )
        self._callback = callback

    def run_once(self):
        """This method has to be called for the callback to be executed."""
        vals = [slider.get_value() for slider in self._sliders]
        self._callback(vals)

    def remove_gui_object(self):
        for slider in self._sliders:
            slider.remove()

    def get_sliders(self) -> List[TkinterGUISlider]:
        return self._sliders


class TkGUIButtonCallback(TkGUICallback):
    """Create a button in Tk and specify the callback to call when button is pressed."""

    def __init__(
        self,
        button_name: str,
        callback: Callable[[], None],
        tk_master: TkinterWidgetMaster = None,
    ):
        """Create a button in Tk and specify the callback to call when button is pressed.

        Args:
            button_name (str): Name of button.
            callback (Callable[[], None]): Handle to a function that should be executed when
                the button is pressed.
            cid (int): The client id to connect to the physics simulator.
        """
        self._button = TkinterGUIButton(name=button_name, tk_master=tk_master)
        self._callback = callback

    def run_once(self):
        """This method has to be called for checking button press and for the
        callback to be executed."""
        if self._button.was_pressed():
            self._callback()

    def remove_gui_object(self):
        self._button.remove()
