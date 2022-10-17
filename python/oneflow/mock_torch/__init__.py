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
from types import ModuleType
from typing import Any
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec
import sys


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


class OneflowImporter(MetaPathFinder, Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("torch"):  # don't touch modules other than torch
            return ModuleSpec(fullname, self)
        return None

    def find_module(self, fullname, path=None):
        spec = self.find_spec(fullname, path)
        return spec

    def create_module(self, spec):
        oneflow_mod_fullname = "oneflow" + spec.name[len("torch") :]
        # get actual oneflow module
        real_spec = find_spec(oneflow_mod_fullname)
        if real_spec is None:
            raise NotImplementedError(oneflow_mod_fullname + error_msg)
        real_mod = module_from_spec(real_spec)
        if sys.modules.get(oneflow_mod_fullname) is None:
            # oneflow/__init__.py can't be executed twice
            real_spec.loader.exec_module(real_mod)
        else:
            real_mod = sys.modules[oneflow_mod_fullname]
        return real_mod

    def exec_module(self, module):
        fullname = "torch" + module.__name__[len("oneflow") :]
        sys.modules[fullname] = ModuleWrapper(module)
        globals()[fullname] = ModuleWrapper(module)


# dynamically mock torch and its submodules
def mock():
    if sys.modules.get("torch") is not None:
        print("Warning: Detected imported torch modules, quitting `mock`")
    else:
        sys.meta_path.insert(0, OneflowImporter())
