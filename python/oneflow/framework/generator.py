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
import oneflow
import oneflow._oneflow_internal

Generator = oneflow._oneflow_internal.Generator


def manual_seed(seed):
    oneflow._oneflow_internal.manual_seed(seed)


def _manual_seed(self, seed):
    self.set_current_seed(seed)
    return self


Generator.manual_seed = _manual_seed


def _getstate(self):
    return {"device": str(self.device), "state": self.get_state()}


def _setstate(self, state_dict):
    self.__init__(state_dict["device"])
    self.set_state(state_dict["state"])


def get_rng_state():
    """
    returns the state of the default random number generator
    """
    return oneflow.default_generator.get_state()


def set_rng_state(state):
    """
    sets the state of the default random number generator to the given state
    """
    return oneflow.default_generator.set_state(state)


default_generator = oneflow._oneflow_internal.default_generator("cpu")
Generator.__getstate__ = _getstate
Generator.__setstate__ = _setstate
