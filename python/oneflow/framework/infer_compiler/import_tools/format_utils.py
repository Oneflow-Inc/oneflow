from typing import Union
from types import FunctionType
import inspect


class MockEntityNameFormatter:
    def __init__(self, prefix: str = "mock_", suffix: str = "_oflow"):
        self.prefix = prefix
        self.suffix = suffix

    def _format_pkg_name(self, pkg_name: str) -> str:
        if pkg_name.startswith(self.prefix) and pkg_name.endswith(self.suffix):
            return pkg_name
        return self.prefix + pkg_name + self.suffix

    def _reverse_pkg_name(self, pkg_name: str) -> str:
        assert pkg_name.startswith(self.prefix) and pkg_name.endswith(
            self.suffix
        ), f"Package name must start with {self.prefix} and end with {self.suffix}, but got {pkg_name}"
        return pkg_name[len(self.prefix) : -len(self.suffix)]

    def _format_full_class_name(self, obj: Union[str, type, FunctionType]):
        if isinstance(obj, type):
            obj = f"{obj.__module__}.{obj.__qualname__}"

        elif isinstance(obj, FunctionType):
            module = inspect.getmodule(obj)
            obj = f"{module.__name__}.{obj.__qualname__}"

        assert isinstance(obj, str), f"obj must be str, but got {type(obj)}"

        if "." in obj:
            pkg_name, cls_name = obj.split(".", 1)
            return f"{self._format_pkg_name(pkg_name)}.{cls_name}"
        else:
            return self._format_pkg_name(obj)

    def format(self, entity: Union[str, type, FunctionType]) -> str:
        return self._format_full_class_name(entity)

    def unformat(self, mock_entity_name: str) -> str:
        if "." in mock_entity_name:
            pkg_name, cls_name = mock_entity_name.split(".", 1)
            return f"{self._reverse_pkg_name(pkg_name)}.{cls_name}"
        else:  # mock_entity_name is a pkg_name
            return self._reverse_pkg_name(mock_entity_name)
