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
from importlib_metadata import version
from .log_utils import logger


def get_support_message():
    recipient_email = "caishenghang@oneflow.org"

    message = f"""\033[91m Advanced features cannot be used !!! \033[0m
If you need unrestricted multiple resolution, quantization support or any other more advanced features, please send an email to \033[91m{recipient_email}\033[0m and tell us about your use case, deployment scale and requirements.
        """
    return message


def is_quantization_enabled():
    import oneflow

    if version("oneflow") < "0.9.1":
        RuntimeError(
            "onediff_comfy_nodes requires oneflow>=0.9.1 to run.", get_support_message()
        )
        return False
    try:
        import diffusers_quant
    except ImportError as e:
        logger.warning(
            f"Failed to import diffusers_quant, Error message: {e}, {get_support_message()}"
        )
        return False
    return hasattr(oneflow._C, "dynamic_quantization")


def is_community_version():
    is_community = not is_quantization_enabled()
    return is_community
