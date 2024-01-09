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
from types import ModuleType

__all__ = ["MockModuleDict", "DummyModule"]


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
        """mock key thorugh value."""
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

    def inverse_name(self, s: str):  # s: spec.name
        return self.inverse[s.split(".")[0]] + s[len(s.split(".")[0]) :]

    def forward_name(self, s: str):
        return self.forward[s.split(".")[0]] + s[len(s.split(".")[0]) :]


class DummyModule(ModuleType):
    def __init__(self, name, verbose=False):
        super().__init__(name)
        self._verbose = verbose

    def __getattr__(self, name):
        if self._verbose:
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

        return DummyModule(self.__name__ + "." + name, self._verbose)

    def __getitem__(self, name):
        new_name = f"{self.__name__}[{name}]"
        if isinstance(name, int):
            if self._verbose:
                print(
                    f'"{self.__name__}" is a dummy object, and `{new_name}` is called. Raising an IndexError to simulate an empty list.'
                )
            raise IndexError
        if self._verbose:
            print(f'"{self.__name__}" is a dummy object, and `{new_name}` is called.')
        return DummyModule(new_name, self._verbose)

    def __call__(self, *args, **kwargs):
        new_name = f'{self.__name__}({", ".join(map(repr, args))}, {", ".join(["{}={}".format(k, repr(v)) for k, v in kwargs.items()])})'
        if self._verbose:
            print(f'"{self.__name__}" is a dummy object, and `{new_name}` is called.')
        return DummyModule(new_name, self._verbose)

    def __bool__(self):
        if self._verbose:
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
