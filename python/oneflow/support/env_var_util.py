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
import os


def string_to_bool(env_str):
    if env_str.lower() in ("1", "true", "yes", "on", "y"):
        return True
    return False


def parse_boolean_from_env(env_var, defalut_value):
    # This function aligns with ParseBooleanFromEnv() in oneflow/core/common/util.cpp
    assert isinstance(env_var, str), "env variable must be string, but got: " + type(
        env_var
    )
    assert isinstance(
        defalut_value, bool
    ), "env variable defalut value must be boolean, but got: " + type(defalut_value)
    if os.getenv(env_var) is None:
        return defalut_value
    else:
        return string_to_bool(os.getenv(env_var))
