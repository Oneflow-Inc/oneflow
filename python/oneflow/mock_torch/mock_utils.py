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
import sys
import sysconfig
import pkgutil
from collections import deque
from importlib import import_module

if sys.version_info < (3, 8):
    try:
        from importlib_metadata import requires
    except ImportError:
        import subprocess

        subprocess.check_call("pip install importlib_metadata", shell=True)
        subprocess.check_call("pip install packaging", shell=True)
else:
    from importlib.metadata import requires

from packaging.requirements import Requirement
from pathlib import Path
from functools import lru_cache
from typing import List, Optional
from types import ModuleType


__all__ = ["MockEnableDisableMixin"]


class PackageDependencyMixin:
    """Get all dependencies of a package filtered by a list of dependencies.

    Example:
        >>> import diffusers #  version 0.24.0
        >>> op = PackageDependencyMixin()
        >>> result = op.has_dependencies("diffusers", ["torch"])
        >>> print(result)
        ['huggingface_hub', 'diffusers']
    """

    pkg_cache = {}  # {pkg: [deps]}

    @staticmethod
    def find_matching_dependencies(
        main_pkg: str, dependencies: List[str], max_visits=1000
    ) -> List[str]:
        @lru_cache()
        def python_stdlib_packages():
            # current python stdlib path
            stdlib_path = sysconfig.get_paths()["stdlib"]

            # use pkgutil to list all modules in the standard library
            python_modules = [
                name for _, name, _ in pkgutil.iter_modules([stdlib_path])
            ]

            # combine built-in module names and Python modules
            all_modules = list(sys.builtin_module_names) + python_modules

            return all_modules

        def format_package_name(pkg: str):
            return Requirement(pkg).name.replace("-", "_")

        @lru_cache()
        def get_requirements(pkg: str):

            python_modules = python_stdlib_packages()
            if pkg in python_modules:
                return []
            try:
                direct_dependencies = requires(pkg)
                if len(direct_dependencies) == 0:
                    return []

                result = set()
                for pkg in direct_dependencies:
                    pkg = format_package_name(pkg)
                    if pkg == main_pkg:
                        continue

                    if pkg not in python_modules:
                        result.add(pkg)

                return list(result)

            except:
                return []

        def is_leaf_package(pkg) -> bool:
            if pkg in dependencies:
                return True

            return len(get_requirements(pkg)) == 0

        main_pkg = format_package_name(main_pkg)

        # build graph
        graph = {}  # {dep: [pkg1, pkg2, ...]}
        queue = deque([main_pkg])
        visited = set()
        stops = set()
        while queue:
            pkg = queue.popleft()
            if is_leaf_package(pkg):
                stops.add(pkg)
                continue
            if pkg in visited:
                continue
            visited.add(pkg)
            if len(visited) > max_visits:
                print(
                    f"\033[1;33mWARNING: max_visits {max_visits} reached, stop searching.\033[0m"
                )
                break

            for req in get_requirements(pkg):
                graph.setdefault(req, set()).add(pkg)
                queue.append(req)

        # init cache and queue
        cache = {}
        visited.clear()
        queue = deque(stops)
        for pkg in stops:
            cache[pkg] = True if pkg in dependencies else False

        # bfs_from_stops
        while queue:
            pkg = queue.popleft()
            if pkg in visited:
                continue
            visited.add(pkg)

            for dep in graph.get(pkg, set()):
                is_ok = cache.get(dep, False)
                if cache[pkg] or is_ok:
                    is_ok = True
                cache[dep] = is_ok
                queue.append(dep)

        return [pkg for pkg, is_ok in cache.items() if is_ok]

    @staticmethod
    def varify_input(main_pkg, dependencies, callback, verbose=False):
        try:
            requires(main_pkg)
        except:
            if verbose:
                print(
                    f"WARNING: main_pkg {main_pkg} has no meta information, please check if it is a valid package."
                )
                print("will set it as its own dependency to avoid error.")
            PackageDependencyMixin.pkg_cache[main_pkg] = [main_pkg] + dependencies

        if not isinstance(main_pkg, str):
            raise ValueError("main_pkg must be a string.")
        if not isinstance(dependencies, list):
            raise ValueError("dependencies must be a list.")
        if not all([isinstance(dep, str) for dep in dependencies]):
            raise ValueError("dependencies must be a list of strings.")
        if callback is not None and not callable(callback):
            raise ValueError("callback must be a callable.")

    @classmethod
    def has_dependencies(
        self,
        main_pkg: str,
        dependencies: List[str],
        callback: callable = None,
        *,
        verbose=False,
    ) -> List[str]:
        """Check if a package has any dependencies in a list of dependencies."""
        PackageDependencyMixin.varify_input(main_pkg, dependencies, callback, verbose)

        deps = PackageDependencyMixin.pkg_cache.get(main_pkg, None)
        if deps is None:
            deps = PackageDependencyMixin.find_matching_dependencies(
                main_pkg, dependencies
            )
            PackageDependencyMixin.pkg_cache.update({main_pkg: deps})

        if verbose:
            print("PackageDependencyMixin : main_pkg=", main_pkg, ", deps=", deps)

        if callback:
            return callback(deps)
        else:
            return deps


class VersionMixin:
    version_cache = {}

    def mock_version(self, module_a: ModuleType, module_b: ModuleType):
        """Mock the version of module_a with the version of module_b."""
        if isinstance(module_a, str):
            module_a = import_module(module_a)
        if isinstance(module_b, str):
            module_b = import_module(module_b)

        attr_name = "__version__"
        orig_attr = getattr(module_a, attr_name, None)
        setattr(module_a, attr_name, getattr(module_b, attr_name, None))
        VersionMixin.version_cache.update({module_a: (attr_name, orig_attr)})

    def restore_version(self):
        for module, (attr_name, orig_attr) in self.version_cache.items():
            setattr(module, attr_name, orig_attr)
        VersionMixin.version_cache.clear()


class MockEnableDisableMixin(PackageDependencyMixin, VersionMixin):
    """Mock torch package using  OneFlow."""

    # list of hazardous modules that may cause issues, handle with care
    hazard_list = [
        "_distutils_hack",
        "importlib",
        "regex",
        "tokenizers",
        "safetensors._safetensors_rust",
    ]

    def is_safe_module(self, module_key):
        k = module_key
        hazard_list = MockEnableDisableMixin.hazard_list

        name = k if "." not in k else k[: k.find(".")]
        if name in hazard_list or k in hazard_list:
            return False
        return True

    def mock_enable(
        self,
        globals,  # parent_globals
        module_dict,  # MockModuleDict object
        *,
        main_pkg: Optional[str] = None,
        mock_version: Optional[str] = None,
        required_dependencies: List[str],
        from_cli=False,
        verbose=False,
        **kwargs,
    ):
        """Mock torch package using  OneFlow.

        Args:
            `globals`: The globals() of the parent module.

            `module_dict`:  MockModuleDict object.

            `main_pkg`: The main package to mock.

            `required_dependencies`: The dependencies to mock for the `main_pkg`.
        """
        if mock_version:
            mock_map = module_dict.forward
            for pkg, mock_pkg in mock_map.items():
                self.mock_version(pkg, mock_pkg)

        if not hasattr(self, "enable_mod_cache"):
            self.enable_mod_cache = {}
        if not hasattr(self, "disable_mod_cache"):
            self.disable_mod_cache = {}
        if not hasattr(self, "mock_safety_packages"):
            self.mock_safety_packages = set()

        if main_pkg:
            # Analyze the dependencies of the main package
            cur_sys_modules = sys.modules.copy()
            existing_deps = self.has_dependencies(
                main_pkg,
                dependencies=required_dependencies,
                callback=lambda x: [dep for dep in x if dep in cur_sys_modules],
                verbose=verbose,
            )
            if verbose:
                print(
                    "Existing dependencies of ",
                    "main_pkg: ",
                    main_pkg,
                    "existing_deps: ",
                    existing_deps,
                )

            self.mock_safety_packages.update(existing_deps)

        # disable non-safe modules loaded before mocking
        def can_disable_mod_cache(k):  # module_key
            if not self.is_safe_module(k):
                return False
            if module_dict.in_forward_dict(k):
                return True
            for dep_pkg in self.mock_safety_packages:
                if k.startswith(dep_pkg + ".") or k == dep_pkg:
                    return True
            return False

        for k, v in sys.modules.copy().items():
            exclude_torch_from_cli = not (from_cli and k == "torch")
            if not exclude_torch_from_cli:  # torch is imported from CLI
                continue

            if can_disable_mod_cache(k):
                aliases = [alias for alias, value in globals.items() if value is v]
                self.disable_mod_cache.update({k: (v, aliases)})
                del sys.modules[k]
                for alias in aliases:
                    del globals[alias]

        # restore modules loaded during mocking
        for k, (v, aliases) in self.enable_mod_cache.items():
            sys.modules.update({k: v})
            for alias in aliases:
                globals.update({alias: v})

    def mock_disable(self, globals, module_dict, delete_list, from_cli=False):
        """Disable the mocked packages."""
        if not hasattr(self, "enable_mod_cache") or not hasattr(
            self, "disable_mod_cache"
        ):
            RuntimeError("Please call mock_enable() first.")

        # disable modules loaded during mocking
        def can_enable_mod_cache(k):  # module_key
            if not self.is_safe_module(k):
                return False
            if module_dict.in_forward_dict(k):
                return True
            return k in delete_list

        for k, v in sys.modules.copy().items():
            if can_enable_mod_cache(k):
                aliases = [alias for alias, value in globals.items() if value is v]
                self.enable_mod_cache.update({k: (v, aliases)})
                del sys.modules[k]
                for alias in aliases:
                    del globals[alias]

        # restore modules loaded during before mocking
        for k, (v, aliases) in self.disable_mod_cache.items():
            sys.modules.update({k: v})
            for alias in aliases:
                globals.update({alias: v})

        if from_cli:
            torch_env = Path(__file__).parent
            sys.path.remove(str(torch_env))

        self.restore_version()
