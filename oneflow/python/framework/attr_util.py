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


def SetAttrValue(attr_value, py_value, default_attr_value):
    if default_attr_value.HasField("at_bool"):
        if py_value is None:
            py_value = True
        assert type(py_value) is bool
        attr_value.at_bool = py_value
    elif default_attr_value.HasField("at_int64"):
        assert type(py_value) is int
        attr_value.at_int64 = py_value
    elif default_attr_value.HasField("at_double"):
        assert type(py_value) is float
        attr_value.at_double = py_value
    elif default_attr_value.HasField("at_string"):
        assert type(py_value) is str
        attr_value.at_string = py_value
    else:
        raise ValueError(
            "config with type %s is invalid. supported types: [bool, int, float, str]"
            % type(py_value)
        )
