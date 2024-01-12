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


def parse_boolean_from_env(env_var, default_value):
    env_var = os.getenv(env_var)
    if env_var is None:
        return default_value
    env_var = env_var.lower()
    return env_var in ("1", "true", "yes", "on", "y")


def set_boolean_env_var(env_var: str, val: bool):
    os.environ[env_var] = str(val)
