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
from functools import partial
import types
from inspect import ismodule, currentframe
from types import ModuleType
from typing import Any, Dict, Optional
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec
from importlib.util import find_spec, module_from_spec
import sys
from typing import List
from zipimport import zipimporter

import oneflow.support.env_var_util as env_var_util
from .mock_modules import MockModuleDict, DummyModule
from .mock_utils import MockEnableDisableMixin


error_msg = """ is not implemented, please submit an issue at  
'https://github.com/Oneflow-Inc/oneflow/issues' including the log information of the error, the 
minimum reproduction code, and the system information."""


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


# module wrapper with checks for existence of methods
class ModuleWrapper(ModuleType):
    # TODO add selcted methods
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
                return DummyModule(new_name, verbose=_importer.verbose)
            else:
                if _importer.lazy and _importer.verbose:
                    print(f"hasattr({self.module.__name__}, {name}) returns False")
                raise AttributeError(new_name + error_msg)
        attr = getattr(self.module, name)
        if ismodule(attr):
            return ModuleWrapper(attr)
        else:
            return attr


class OneflowImporter(MockEnableDisableMixin, MetaPathFinder, Loader):
    def __init__(self):
        # module_from_spec will try to call the loader's create_module, resulting in infinite recursion
        self.in_create_module = False
        self.enable = False
        # both __init__.py of oneflow and torch can't be executed multiple times, so we use a cache
        self.enable_mod_cache = {}
        self.disable_mod_cache = {}
        # Record modules loaded during mocking for deletion
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
                oneflow_mod_fullname = module_dict_global.forward_name(spec.name)
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
                        return DummyModule(oneflow_mod_fullname, verbose=self.verbose)
                    else:
                        raise ModuleNotFoundError(oneflow_mod_fullname + error_msg)

                real_mod = module_from_spec(real_spec)
                loader = real_spec.loader
                if isinstance(loader, zipimporter):
                    # TODO: verify can mock torch as oneflow in zipimporter
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
        module_name = module.__name__
        if module_dict_global.in_inverse_dict(module_name):
            fullname = module_dict_global.inverse_name(module_name)
        if self.enable:
            if not isinstance(module, DummyModule):
                module = ModuleWrapper(module)
        sys.modules[fullname] = module
        globals()[fullname] = module

    def _enable(
        self,
        globals=None,
        lazy=False,
        verbose=False,
        *,
        main_pkg: str = None,
        mock_version: bool = None,
        required_dependencies: List[str] = [],
        from_cli: bool = False,
    ):

        if verbose:
            print("enable mock torch", globals["__name__"])

        if self.enable:  # already enabled
            of_importer_module_name = self.globals["__name__"]
            input_module_name = globals["__name__"]
            if of_importer_module_name != input_module_name:
                print(
                    f"Warning: {of_importer_module_name} is already enabled, but {input_module_name} is trying to enable it again. skip."
                )
            return

        # record config for re-enabling
        self._mock_enable_config = {k: v for k, v in locals().items() if k != "self"}
        # insert importer to the first place of meta_path
        sys.meta_path.insert(0, self)

        self.lazy = lazy
        self.verbose = verbose
        self.from_cli = from_cli
        self.globals = globals

        self.mock_enable(
            globals=globals,
            module_dict=module_dict_global,
            main_pkg=main_pkg,
            mock_version=mock_version,
            required_dependencies=required_dependencies,
            from_cli=from_cli,
            verbose=verbose,
        )
        self.enable = True

    def _disable(self, globals, *, verbose=False):
        if verbose:
            print(
                "disable mock torch in",
                globals["__name__"],
                "\tself.enable: ",
                self.enable,
            )

        if not self.enable:  # already disabled
            return

        of_importer_module_name = self.globals["__name__"]
        input_module_name = globals["__name__"]
        if of_importer_module_name != input_module_name:
            raise RuntimeError(
                f"Error: {of_importer_module_name} is enabled, but {input_module_name} is trying to disable it. must disable it in the same module."
            )

        self.mock_disable(
            globals=globals,
            module_dict=module_dict_global,
            delete_list=self.delete_list,
            from_cli=self.from_cli,
        )

        sys.meta_path.remove(self)
        self.enable = False
        self.delete_list = []
        self.globals = None


_importer = OneflowImporter()


class BaseMockConfig:
    def __init__(
        self,
        lazy: Optional[bool] = None,
        verbose: Optional[bool] = None,
        extra_dict: Optional[Dict[str, str]] = None,
        *,
        main_pkg: Optional[str] = None,
        mock_version: Optional[str] = None,
        required_dependencies: List[str] = [],
        _from_cli: bool = False,
    ):
        global module_dict_global
        module_dict_global = MockModuleDict(extra_dict)
        module_dict_global.add("torch", "oneflow")

        required_dependencies.extend(
            [k for k in extra_dict or {} if k not in required_dependencies]
        )
        if "torch" not in required_dependencies:
            required_dependencies.append("torch")

        parse_bool_env = partial(
            env_var_util.parse_boolean_from_env, defalut_value=False
        )

        forcedly_disabled_by_env_var = parse_bool_env("ONEFLOW_DISABLE_MOCK_TORCH")

        lazy = lazy if lazy is not None else parse_bool_env("ONEFLOW_MOCK_TORCH_LAZY")
        verbose = (
            verbose
            if verbose is not None
            else parse_bool_env("ONEFLOW_MOCK_TORCH_VERBOSE")
        )

        self.lazy = lazy
        self.verbose = verbose
        self.forcedly_disabled_by_env_var = forcedly_disabled_by_env_var
        self.required_dependencies = required_dependencies
        self.parse_bool_env = parse_bool_env
        self._from_cli = _from_cli
        self.main_pkg = main_pkg
        self.mock_version = mock_version


class enable(BaseMockConfig):
    """https://docs.oneflow.org/master/cookies/oneflow_torch.html"""

    def __init__(
        self,
        lazy: Optional[bool] = None,
        verbose: Optional[bool] = None,
        extra_dict: Optional[Dict[str, str]] = None,
        *,
        main_pkg: Optional[str] = None,
        mock_version: Optional[str] = None,
        required_dependencies: List[str] = [],
        _from_cli: bool = False,
    ):
        super().__init__(
            lazy=lazy,
            verbose=verbose,
            extra_dict=extra_dict,
            main_pkg=main_pkg,
            mock_version=mock_version,
            required_dependencies=required_dependencies,
            _from_cli=_from_cli,
        )

        if self.forcedly_disabled_by_env_var:  # super().__init__ will set this
            return

        self.globals = currentframe().f_back.f_globals
        self.skip_processing = False
        if getattr(_importer, "globals", None) is not None:
            import_name = _importer.globals["__name__"]
            if import_name == self.globals["__name__"]:
                self.skip_processing = True
                return

        self._importer_enable = _importer.enable
        if self._importer_enable:
            self._mock_enable_config = _importer._mock_enable_config
            _importer._disable(_importer.globals, verbose=self.verbose)

        _importer._enable(
            self.globals,
            lazy,
            verbose,
            main_pkg=main_pkg,
            mock_version=mock_version,
            required_dependencies=required_dependencies,
            from_cli=_from_cli,
        )

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):

        if self.forcedly_disabled_by_env_var or self.skip_processing:
            return

        _importer._disable(_importer.globals, verbose=self.verbose)

        if self._importer_enable:
            _importer._enable(
                # When re-enabling mock torch, from_cli shoule always be False
                **self._mock_enable_config,
            )


class disable:
    def __init__(self):
        self._importer_enable = _importer.enable
        if not self._importer_enable:
            return

        self.globals = currentframe().f_back.f_globals
        self.lazy = _importer.lazy
        self.verbose = _importer.verbose
        self._mock_enable_config = _importer._mock_enable_config
        _importer._disable(_importer.globals, verbose=self.verbose)

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        if self._importer_enable:
            _importer._enable(
                # When re-enabling mock torch, from_cli shoule always be False
                **self._mock_enable_config,
            )


def is_enabled():
    return _importer.enable
