class PyRCFExceptionBase(Exception): ...


# CtrlLoop Signal
class CtrlLoopExitSignal(KeyboardInterrupt):
    """Request to exit the control loop."""


# UI Errors
class UIException(IOError):
    """Exceptions relating to user interfaces (UIBase objects)."""


class NotConnectedError(UIException):
    """Exception to be thrown when a UI interface could not be detected or was disconnected."""
