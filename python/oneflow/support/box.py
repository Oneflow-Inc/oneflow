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


class Box(object):
    def __init__(self, *arg):
        assert len(arg) <= 1
        self.has_value_ = len(arg) > 0
        self.value_ = None
        if self.has_value_:
            self.value_ = arg[0]

    @property
    def value(self):
        assert self.has_value_
        return self.value_

    @property
    def value_setter(self):
        return lambda val: self.set_value(val)

    def set_value(self, val):
        self.value_ = val
        self.has_value_ = True

    def has_value(self):
        return self.has_value_
