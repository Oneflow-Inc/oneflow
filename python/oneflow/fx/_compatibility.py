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
from typing import Any, Dict
import textwrap

_BACK_COMPAT_OBJECTS: Dict[Any, None] = {}
_MARKED_WITH_COMATIBLITY: Dict[Any, None] = {}


def compatibility(is_backward_compatible: bool):
    if is_backward_compatible:

        def mark_back_compat(fn):
            docstring = textwrap.dedent(getattr(fn, "__doc__", None) or "")
            docstring += """
.. note::
    Backwards-compatibility for this API is guaranteed.
"""
            fn.__doc__ = docstring
            _BACK_COMPAT_OBJECTS.setdefault(fn)
            _MARKED_WITH_COMATIBLITY.setdefault(fn)
            return fn

        return mark_back_compat
    else:

        def mark_not_back_compat(fn):
            docstring = textwrap.dedent(getattr(fn, "__doc__", None) or "")
            docstring += """
.. warning::
    This API is experimental and is *NOT* backward-compatible.
"""
            fn.__doc__ = docstring
            _MARKED_WITH_COMATIBLITY.setdefault(fn)
            return fn

        return mark_not_back_compat
