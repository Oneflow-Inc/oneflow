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
import oneflow
from typing import Any
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec
import sys

__path__ = oneflow.__path__
error_msg = """ is not implemented, please submit issues in 
'https://github.com/Oneflow-Inc/oneflow/issues' include the log information of the error, the 
minimum reproduction code, and the system information."""

# module wrapper with checks for existence of methods
class ModuleWrapper(ModuleType):
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name: str) -> Any:
        if not hasattr(self.module, name):
            if name == "__path__":  # fix: from somemod import attr
                return None
            if name == "__all__":
                setattr(self.module, name, dir(self.module))
                return getattr(self.module, name)
            raise NotImplementedError(self.module.__name__ + "." + name + error_msg)
        attr = getattr(self.module, name)
        if ismodule(attr):
            return ModuleWrapper(attr)
        else:
            return attr


def __getattr__(name: str):
    return ModuleWrapper(oneflow).__getattr__(name)


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
        real_module = module_from_spec(real_spec)
        real_spec.loader.exec_module(real_module)
        return real_module

    def exec_module(self, module):
        fullname = "torch" + module.__name__[len("oneflow") :]
        sys.modules[fullname] = ModuleWrapper(module)
        globals()[fullname] = ModuleWrapper(module)


# register importer in meta path
sys.meta_path.insert(0, OneflowImporter())
