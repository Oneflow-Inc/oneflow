"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from inspect import ismodule
import importlib
from contextlib import contextmanager
from types import ModuleType
from typing import Dict, List
from .mock_importer import enable


class DynamicMockModule(ModuleType):
    def __init__(
        self, pkg_name: str, obj_entity: ModuleType, main_pkg_enable: callable,
    ):
        self._pkg_name = pkg_name
        self._obj_entity = obj_entity  # ModuleType or _LazyModule
        self._main_pkg_enable = main_pkg_enable
        self._intercept_dict = {}

    def __repr__(self) -> str:
        return f"<DynamicMockModule {self._pkg_name} {self._obj_entity}>"

    def hijack(self, module_name: str, obj: object):
        self._intercept_dict[module_name] = obj

    @classmethod
    def from_package(
        cls,
        main_pkg: str,
        *,
        lazy: bool = True,
        verbose: bool = False,
        extra_dict: Dict[str, str] = None,
        required_dependencies: List[str] = [],
    ):
        assert isinstance(main_pkg, str)

        @contextmanager
        def main_pkg_enable():
            with enable(
                lazy=lazy,
                verbose=verbose,
                extra_dict=extra_dict,
                main_pkg=main_pkg,
                mock_version=True,
                required_dependencies=required_dependencies,
            ):
                yield

        with main_pkg_enable():
            obj_entity = importlib.import_module(main_pkg)
        return cls(main_pkg, obj_entity, main_pkg_enable)

    def _get_module(self, _name: str):
        # Fix Lazy import
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/__init__.py#L728-L734
        module_name = f"{self._obj_entity.__name__}.{_name}"
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __getattr__(self, name: str):
        fullname = f"{self._obj_entity.__name__}.{name}"
        if fullname in self._intercept_dict:
            return self._intercept_dict[fullname]

        with self._main_pkg_enable():
            obj_entity = getattr(self._obj_entity, name, None)
            if obj_entity is None:
                obj_entity = self._get_module(name)

        if ismodule(obj_entity):
            return DynamicMockModule(self._pkg_name, obj_entity, self._main_pkg_enable)

        return obj_entity

    def __all__(self):
        with self._main_pkg_enable():
            return dir(self._obj_entity)
