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
from typing import Any
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec
import sys
from contextlib import contextmanager

_first_init = True

error_msg = """ is not implemented, please submit an issue at  
'https://github.com/Oneflow-Inc/oneflow/issues' including the log information of the error, the 
minimum reproduction code, and the system information."""

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
            raise NotImplementedError(self.module.__name__ + "." + name + error_msg)
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

    def find_spec(self, fullname, path, target=None):
        if _is_torch(fullname):  # don't touch modules other than torch
            # for first import of real torch, we use default meta path finders, not our own
            if not self.enable and self.disable_mod_cache.get(fullname) is None:
                return None
            return ModuleSpec(fullname, self)
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
                real_spec = find_spec(oneflow_mod_fullname)
                if real_spec is None:
                    raise NotImplementedError(oneflow_mod_fullname + error_msg)
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
            module = ModuleWrapper(module)
        sys.modules[fullname] = module
        globals()[fullname] = module

    def _enable(self, globals):
        global _first_init
        if _first_init:
            _first_init = False
            self.enable = False  # deal with previously imported torch
            sys.meta_path.insert(0, self)
            self._enable(globals)
            return
        if self.enable:  # already enabled
            return
        for k, v in sys.modules.copy().items():
            if _is_torch(k):
                self.disable_mod_cache.update({k: v})
                del sys.modules[k]
                try:
                    del globals[k]
                except KeyError:
                    pass
        for k, v in self.enable_mod_cache.items():
            sys.modules.update({k: v})
            globals.update({k: v})
        self.enable = True

    def _disable(self, globals):
        if not self.enable:  # already disabled
            return
        for k, v in sys.modules.copy().items():
            if _is_torch(k):
                self.enable_mod_cache.update({k: v})
                del sys.modules[k]
                try:
                    del globals[k]
                except KeyError:
                    pass
        for k, v in self.disable_mod_cache.items():
            sys.modules.update({k: v})
            globals.update({k: v})
        self.enable = False


_importer = OneflowImporter()


class enable:
    def __init__(self):
        self.enable = _importer.enable
        if self.enable:
            return
        globals = currentframe().f_back.f_globals
        self.globals = globals
        _importer._enable(globals)

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
        _importer._disable(globals)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        if self.enable:
            _importer._enable(self.globals)
