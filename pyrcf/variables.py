import importlib
import importlib.util


def package_available(package_name: str):
    return importlib.util.find_spec(package_name) is not None


# optional third party libraries
TORCH_AVAILABLE = package_available("torch")
MUJOCO_ROBOT_AVAILABLE = package_available("mujoco_robot")


__all__ = ["MUJOCO_ROBOT_AVAILABLE", "TORCH_AVAILABLE"]
