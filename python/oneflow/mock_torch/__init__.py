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
import builtins
import types
from inspect import ismodule, currentframe
from types import ModuleType
from typing import Any, Dict, Optional
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec
import sys
import os
from pathlib import Path
from contextlib import contextmanager
from zipimport import zipimporter

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

# patch hasattr so that
# 1. torch.not_exist returns DummyModule object, but
# 2. hasattr(torch, "not_exist") still returns False
_builtin_hasattr = builtins.hasattr
if not isinstance(_builtin_hasattr, types.BuiltinFunctionType):
    raise Exception("hasattr already patched by someone else!")


def hasattr(obj, name):
    return _builtin_hasattr(obj, name)


builtins.hasattr = hasattr


def probably_called_from_hasattr():
    frame = currentframe().f_back.f_back
    return frame.f_code is hasattr.__code__


class MockModuleDict:
    def __init__(self, mapping=None):
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError("Extra mock library must be a dict.")
        self.forward = {}
        self.inverse = {}
        if mapping is not None:
            for key, value in mapping.items():
                self.add(key, value)

    def add(self, key, value):
        if key in self.forward or value in self.inverse:
            raise ValueError("Key or value already exists.")
        self.forward[key] = value
        self.inverse[value] = key

    def remove(self, key=None, value=None):
        if key is not None:
            value = self.forward.pop(key)
            self.inverse.pop(value)
        elif value is not None:
            key = self.inverse.pop(value)
            self.forward.pop(key)
        else:
            raise ValueError("Must provide a key or value to remove.")

    def in_forward_dict(self, s):
        return s.split(".")[0] in self.forward.keys()

    def in_inverse_dict(self, s):
        return s.split(".")[0] in self.inverse.keys()


# module wrapper with checks for existence of methods
class ModuleWrapper(ModuleType):
    def __init__(self, module):
        self.module = module

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name != "module":
            setattr(self.module, name, value)

    def __getattr__(self, name: str) -> Any:
        if not hasattr(self.module, name):
            if name == "__path__":
                return None
            if name == "__all__":
                return [attr for attr in dir(self.module) if not attr.startswith("_")]
            new_name = self.module.__name__ + "." + name
            if _importer.lazy and not probably_called_from_hasattr():
                if _importer.verbose:
                    print(
                        f'"{new_name}" is not found in oneflow, use dummy object as fallback.'
                    )
                return DummyModule(new_name)
            else:
                if _importer.lazy and _importer.verbose:
                    print(f"hasattr({self.module.__name__}, {name}) returns False")
                raise AttributeError(new_name + error_msg)
        attr = getattr(self.module, name)
        if ismodule(attr):
            return ModuleWrapper(attr)
        else:
            return attr


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
        if module_dict_global.in_forward_dict(
            fullname
        ):  # don't touch modules other than torch or extra libs module
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
            if module_dict_global.in_forward_dict(spec.name):
                oneflow_mod_fullname = (
                    module_dict_global.forward[spec.name.split(".")[0]]
                    + spec.name[len(spec.name.split(".")[0]) :]
                )
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
                loader = real_spec.loader
                if isinstance(loader, zipimporter):
                    pass
                else:
                    loader.exec_module(real_mod)
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
        if module_dict_global.in_inverse_dict(module.__name__):
            fullname = (
                module_dict_global.inverse[module.__name__.split(".")[0]]
                + module.__name__[len(module.__name__.split(".")[0]) :]
            )
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
            if (not (from_cli and k == "torch")) and module_dict_global.in_forward_dict(
                k
            ):
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
            if module_dict_global.in_forward_dict(k):
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

    def __enter__(self):
        raise RuntimeError(
            f'"{self.__name__}" is a dummy object, and does not support "with" statement.'
        )

    def __exit__(self, exception_type, exception_value, traceback):
        raise RuntimeError(
            f'"{self.__name__}" is a dummy object, and does not support "with" statement.'
        )

    def __subclasscheck__(self, subclass):
        return False

    def __instancecheck__(self, instance):
        return False


class enable:
    def __init__(
        self,
        lazy: Optional[bool] = None,
        verbose: Optional[bool] = None,
        extra_dict: Optional[Dict[str, str]] = None,
        *,
        _from_cli: bool = False,
    ):
        global module_dict_global
        module_dict_global = MockModuleDict(extra_dict)
        module_dict_global.add("torch", "oneflow")
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
        _importer._disable(globals)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        if self.enable:
            _importer._enable(
                # When re-enabling mock torch, from_cli shoule always be False
                self.globals,
                self.lazy,
                self.verbose,
                from_cli=False,
            )


def is_enabled():
    return _importer.enable
