import importlib
import importlib.util


def package_available(package_name: str):
    return importlib.util.find_spec(package_name) is not None


# optional third party libraries
TORCH_AVAILABLE = package_available("torch")


__all__ = ["TORCH_AVAILABLE"]
