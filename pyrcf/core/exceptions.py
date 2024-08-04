class PyRCFExceptionBase(Exception): ...


class CtrlLoopExitSignal(KeyboardInterrupt):
    """Request to exit the control loop."""
