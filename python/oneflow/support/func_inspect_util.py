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
import inspect
import sys

if sys.version_info > (2, 7) and sys.version_info < (3, 0):

    def GetArgNameAndDefaultTuple(func):
        """
      returns a dictionary of arg_name:default_values for the input function
      """
        (args, varargs, keywords, defaults) = inspect.getargspec(func)
        defaults = list(defaults) if defaults is not None else []
        while len(defaults) < len(args):
            defaults.insert(0, None)
        return tuple(zip(args, defaults))


elif sys.version_info >= (3, 0):

    def GetArgNameAndDefaultTuple(func):
        signature = inspect.signature(func)
        return tuple(
            [
                (k, v.default if v.default is not inspect.Parameter.empty else None)
                for (k, v) in signature.parameters.items()
            ]
        )


else:
    raise NotImplementedError


def GetArgDefaults(func):
    return tuple(map(lambda x: x[1], GetArgNameAndDefaultTuple(func)))
