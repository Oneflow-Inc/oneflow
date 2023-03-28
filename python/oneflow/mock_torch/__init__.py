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
from inspect import ismodule, currentframe
from types import ModuleType
from typing import Any, Optional
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec
import sys
import os
from pathlib import Path
from contextlib import contextmanager

import oneflow.support.env_var_util as env_var_util

_first_init = True

error_msg = """ is not implemented, please submit an issue at  
'https://github.com/Oneflow-Inc/oneflow/issues' including the log information of the error, the 
minimum reproduction code, and the system information."""

hazard_list = [
    "_distutils_hack",
    "importlib",
    "regex",
    "tokenizers",
    "safetensors._safetensors_rust",
]

# module wrapper with checks for existence of methods
class ModuleWrapper(ModuleType):
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name: str) -> Any:
        if not hasattr(self.module, name):
            if name == "__path__":
                return None
            if name == "__all__":
                return [attr for attr in dir(self.module) if not attr.startswith("_")]
            new_name = self.module.__name__ + "." + name
            if _importer.lazy:
                blacklist = ["scaled_dot_product_attention"]
                if name in blacklist:
                    if _importer.verbose:
                        print(f'"{new_name}" is in blacklist, raise AttributeError')
                    raise AttributeError(new_name + error_msg)
                else:
                    if _importer.verbose:
                        print(
                            f'"{new_name}" is not found in oneflow, use dummy object as fallback.'
                        )
                    return DummyModule(new_name)
            else:
                raise AttributeError(new_name + error_msg)
        attr = getattr(self.module, name)
        if ismodule(attr):
            return ModuleWrapper(attr)
        else:
            return attr


def _is_torch(s: str):
    return s == "torch" or s.startswith("torch.")


class OneflowImporter(MetaPathFinder, Loader):
    def __init__(self):
        # module_from_spec will try to call the loader's create_module, resulting in infinite recursion
        self.in_create_module = False
        self.enable = False
        # both __init__.py of oneflow and torch can't be executed multiple times, so we use a cache
        self.enable_mod_cache = {}
        self.disable_mod_cache = {}
        self.delete_list = []

    def find_spec(self, fullname, path, target=None):
        if _is_torch(fullname):  # don't touch modules other than torch
            # for first import of real torch, we use default meta path finders, not our own
            if not self.enable and self.disable_mod_cache.get(fullname) is None:
                return None
            return ModuleSpec(fullname, self)
        self.delete_list.append(fullname)
        return None

    def find_module(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        return spec

    def create_module(self, spec):
        if self.in_create_module:
            return None
        self.in_create_module = True
        if self.enable:
            oneflow_mod_fullname = "oneflow" + spec.name[len("torch") :]
            if (
                sys.modules.get(oneflow_mod_fullname) is None
                and self.enable_mod_cache.get(spec.name) is None
            ):
                # get actual oneflow module
                try:
                    real_spec = find_spec(oneflow_mod_fullname)
                except ModuleNotFoundError:
                    real_spec = None
                if real_spec is None:
                    self.in_create_module = False
                    if self.lazy:
                        if self.verbose:
                            print(
                                f"{oneflow_mod_fullname} is not found in oneflow, use dummy object as fallback."
                            )
                        return DummyModule(oneflow_mod_fullname)
                    else:
                        raise ModuleNotFoundError(oneflow_mod_fullname + error_msg)

                real_mod = module_from_spec(real_spec)
                real_spec.loader.exec_module(real_mod)
            else:
                real_mod = sys.modules.get(oneflow_mod_fullname)
                if real_mod is None:
                    real_mod = self.enable_mod_cache[spec.name]
            self.in_create_module = False
            return real_mod
        else:
            torch_full_name = spec.name
            real_mod = self.disable_mod_cache[torch_full_name]
            self.in_create_module = False
            return real_mod

    def exec_module(self, module):
        fullname = "torch" + module.__name__[len("oneflow") :]
        if self.enable:
            if not isinstance(module, DummyModule):
                module = ModuleWrapper(module)
        sys.modules[fullname] = module
        globals()[fullname] = module

    def _enable(self, globals, lazy: bool, verbose: bool, *, from_cli: bool = False):
        global _first_init
        if _first_init:
            _first_init = False
            self.enable = False  # deal with previously imported torch
            sys.meta_path.insert(0, self)
            self._enable(globals, lazy, verbose, from_cli=from_cli)
            return
        self.lazy = lazy
        self.verbose = verbose
        self.from_cli = from_cli
        if self.enable:  # already enabled
            return
        for k, v in sys.modules.copy().items():
            if (not (from_cli and k == "torch")) and _is_torch(k):
                aliases = list(filter(lambda alias: globals[alias] is v, globals))
                self.disable_mod_cache.update({k: (v, aliases)})
                del sys.modules[k]
                for alias in aliases:
                    del globals[alias]
        for k, (v, aliases) in self.enable_mod_cache.items():
            sys.modules.update({k: v})
            for alias in aliases:
                globals.update({alias: v})
        self.enable = True

    def _disable(self, globals):
        if not self.enable:  # already disabled
            return
        for k, v in sys.modules.copy().items():
            if _is_torch(k):
                aliases = list(filter(lambda alias: globals[alias] is v, globals))
                self.enable_mod_cache.update({k: (v, aliases)})
                del sys.modules[k]
                for alias in aliases:
                    del globals[alias]
            name = k if "." not in k else k[: k.find(".")]
            if (
                not name in hazard_list
                and not k in hazard_list
                and k in self.delete_list
            ):
                aliases = list(filter(lambda alias: globals[alias] is v, globals))
                self.enable_mod_cache.update({k: (v, aliases)})
                del sys.modules[k]
        for k, (v, aliases) in self.disable_mod_cache.items():
            sys.modules.update({k: v})
            for alias in aliases:
                globals.update({alias: v})
        if self.from_cli:
            torch_env = Path(__file__).parent
            sys.path.remove(str(torch_env))

        self.enable = False


_importer = OneflowImporter()


class DummyModule(ModuleType):
    def __getattr__(self, name):
        if _importer.verbose:
            print(
                f'"{self.__name__}" is a dummy object, and its attr "{name}" is accessed.'
            )
        if name == "__path__":
            return None
        if name == "__all__":
            return []
        if name == "__file__":
            return None
        if name == "__mro_entries__":
            return lambda x: ()
        return DummyModule(self.__name__ + "." + name)

    def __getitem__(self, name):
        new_name = f"{self.__name__}[{name}]"
        if isinstance(name, int):
            if _importer.verbose:
                print(
                    f'"{self.__name__}" is a dummy object, and `{new_name}` is called. Raising an IndexError to simulate an empty list.'
                )
            raise IndexError
        if _importer.verbose:
            print(f'"{self.__name__}" is a dummy object, and `{new_name}` is called.')
        return DummyModule(new_name)

    def __call__(self, *args, **kwargs):
        new_name = f'{self.__name__}({", ".join(map(repr, args))}, {", ".join(["{}={}".format(k, repr(v)) for k, v in kwargs.items()])})'
        if _importer.verbose:
            print(f'"{self.__name__}" is a dummy object, and `{new_name}` is called.')
        return DummyModule(new_name)

    def __bool__(self):
        if _importer.verbose:
            print(
                f'"{self.__name__}" is a dummy object, and its bool value is accessed.'
            )
        return False


class enable:
    def __init__(
        self,
        lazy: Optional[bool] = None,
        verbose: Optional[bool] = None,
        *,
        _from_cli: bool = False,
    ):
        self.enable = _importer.enable
        forcedly_disabled_by_env_var = env_var_util.parse_boolean_from_env(
            "ONEFLOW_DISABLE_MOCK_TORCH", False
        )
        globals = currentframe().f_back.f_globals
        self.globals = globals
        lazy = (
            lazy
            if lazy is not None
            else env_var_util.parse_boolean_from_env("ONEFLOW_MOCK_TORCH_LAZY", False)
        )
        verbose = (
            verbose
            if verbose is not None
            else env_var_util.parse_boolean_from_env(
                "ONEFLOW_MOCK_TORCH_VERBOSE", False
            )
        )
        if forcedly_disabled_by_env_var:
            return
        _importer._enable(globals, lazy, verbose, from_cli=_from_cli)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        if not self.enable:
            _importer._disable(self.globals)


class disable:
    def __init__(self):
        self.enable = _importer.enable
        if not self.enable:
            return
        globals = currentframe().f_back.f_globals
        self.globals = globals
        self.lazy = _importer.lazy
        self.verbose = _importer.verbose
        self.from_cli = _importer.from_cli
        _importer._disable(globals)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        if self.enable:
            _importer._enable(
                self.globals, self.lazy, self.verbose, from_cli=self.from_cli
            )


def is_enabled():
    return _importer.enable
