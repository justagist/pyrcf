from typing import Callable
import tkinter as tk

from ..time_utils import RateTrigger


class TkinterWidgetMaster:
    """Handles a Tk window (called 'master')."""

    def __init__(self, max_update_rate: float = 100, window_name: str = "Default Tk Master"):
        """Handles a Tk window (called 'master').

        Args:
            max_update_rate (float, optional): Max update rate for the update method.
                Defaults to 100 Hz.
                NOTE: The `update` method has to be called in a loop externally. There
                is no threads defined in this class.
            window_name (str, optional): Title for the window. Defaults to "Default Tk Master".
        """

        self.master_widget = tk.Tk()
        self.master_widget.title(window_name)
        self.master_widget.geometry("750x350+200+200")
        self.trigger = RateTrigger(rate=max_update_rate)

    def update(self):
        """The update method to be called in a loop to refresh all the widgets attached
        to this window. This will call the master update only if the rate specified
        during construction is met.
        """
        if self.trigger():
            self.master_widget.update()


class _COMMON_TK_MASTER:
    """Internal class that creates a singleton instance of TkinterWidgetMaster
    to be used by all widgets when a default master is not specified."""

    __COMMON_TKINTER_MASTER: TkinterWidgetMaster = None

    @staticmethod
    def _get_tkinter_common_master():
        if _COMMON_TK_MASTER.__COMMON_TKINTER_MASTER is None:
            _COMMON_TK_MASTER.__COMMON_TKINTER_MASTER = TkinterWidgetMaster()
        return _COMMON_TK_MASTER.__COMMON_TKINTER_MASTER


class TkinterGUISlider:
    """Create a slider gui in Tkinter."""

    def __init__(
        self,
        name: str,
        lower_lim: float,
        upper_lim: float,
        default_val: float,
        tk_master: TkinterWidgetMaster = None,
        mapping_function: Callable[[float], float] = lambda x: x,
        slider_resolution: float = 0.1,
    ):
        """Create a slider gui in Tkinter.

        Args:
            name (str): Name for slider.
            lower_lim (float): The min value for slider.
            upper_lim (float): The max value for slider.
            default_val (float): Default starting value for slider.
            tk_master (TkinterWidgetMaster, optional): The TkWidgetMaster to use. If None
                provided, will create a new master (shared between all other GUI objects
                created similarly with None).
            mapping_function (Callable[[float], float], optional): Mapping to do for the
                value read from the slider (e.g. convert radian to degrees)
        """
        self.tk_master = (
            tk_master if tk_master is not None else _COMMON_TK_MASTER._get_tkinter_common_master()
        )
        self.slider = tk.Scale(
            master=self.tk_master.master_widget,
            label=name,
            from_=lower_lim,
            to=upper_lim,
            orient=tk.HORIZONTAL,
            resolution=slider_resolution,
        )
        self.slider.set(default_val)
        self.slider.pack(padx=10, fill="both")

        assert callable(mapping_function)
        self._mapping_function = mapping_function

    def get_value(self):
        """Read the value from the slider gui."""
        self.slider.update()
        self.slider.pack(padx=10, fill="both")
        return self._mapping_function(self.slider.get())

    def remove(self, permanent: bool = True):
        """Remove the slider from gui."""
        if permanent:
            self.slider.destroy()
        else:
            self.slider.pack_forget()

    def set_value(self, value: float):
        """Set the value of the slider to specified value.

        Args:
            value (float): Value to set the slider to.
        """
        self.slider.set(value=value)
        self.slider.update()


class TkinterGUIButton:
    """Create a button in Tkinter GUI, and check if it was pressed.
    Meant to be used in a loop."""

    def __init__(
        self,
        name: str,
        tk_master: TkinterWidgetMaster = None,
    ):
        """Create a button in Tkinter GUI, and check if it was pressed.
        Meant to be used in a loop.

            Args:
                name (str): Name on button.
                cid (int): The client id to connect to the physics simulator.
        """
        self.tk_master = (
            tk_master if tk_master is not None else _COMMON_TK_MASTER._get_tkinter_common_master()
        )
        self._prev_val = 0
        self._val = 0
        self.button = tk.Button(
            master=self.tk_master.master_widget,
            text=name,
            command=self._button_press_cb,
        )
        self.button.pack(padx=10, fill="both")
        self.button.update()

    def _button_press_cb(self):
        self._val += 1

    def was_pressed(self) -> bool:
        """Checks if this button was pressed since the last time this method was called."""
        self.button.update()
        triggered = self._val > self._prev_val
        self._prev_val = self._val
        return triggered

    def remove(self, permanent: bool = True):
        """Remove the slider from gui."""
        if permanent:
            self.button.destroy()
        else:
            self.button.pack_forget()
