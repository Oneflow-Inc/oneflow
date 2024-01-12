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
# TODO: remove this file to diffusers/src/infer_compiler_registry/register_diffusers
from abc import ABC, abstractmethod
from .log_utils import logger

try:
    import diffusers
    from diffusers.models.attention_processor import Attention
except ImportError:
    diffusers = None
    logger.warning("diffusers not found, some features will be disabled.")

_IS_DIFFUSERS_AVAILABLE = diffusers is not None


class InstanceChecker(ABC):
    @abstractmethod
    def is_attention_instance(self, instance):
        pass


class DiffusersChecker(InstanceChecker):
    def is_attention_instance(self, instance):
        if not _IS_DIFFUSERS_AVAILABLE:
            return False
        return isinstance(instance, Attention)


diffusers_checker = DiffusersChecker()
