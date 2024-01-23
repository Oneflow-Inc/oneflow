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
import logging
import os
import types
from pathlib import Path
from typing import Dict, List, Union

from oneflow.framework.infer_compiler.import_tools.importer import LazyMocker
from oneflow.framework.infer_compiler.utils.log_utils import logger

__all__ = ["transform_mgr"]


class TransformManager:
    """TransformManager

    __init__ args:
        `debug_mode`: Whether to print debug info.
        `tmp_dir`: The temp dir to store mock files.
    """

    def __init__(self, debug_mode=False, tmp_dir="./output"):
        self.debug_mode = debug_mode
        self._torch_to_oflow_cls_map = {}
        self._setup_logger()
        self.mocker = LazyMocker(prefix="", suffix="", tmp_dir=None)

    def _setup_logger(self):
        name = "ONEDIFF"
        level = logging.DEBUG if self.debug_mode else logging.ERROR
        logger.configure_logging(name=name, file_name=None, level=level, log_dir=None)
        self.logger = logger

    def get_mocked_packages(self):
        return self.mocker.mocked_packages

    def load_class_proxies_from_packages(self, package_names: List[Union[Path, str]]):
        self.logger.debug(f"Loading modules: {package_names}")
        for package_name in package_names:
            self.mocker.mock_package(package_name)
            self.logger.info(f"Loaded Mock Torch Package: {package_name} successfully")

    def update_class_proxies(self, class_proxy_dict: Dict[str, type], verbose=True):
        """Update `_torch_to_oflow_cls_map` with `class_proxy_dict`.

        example:
            `class_proxy_dict = {"mock_torch.nn.Conv2d": flow.nn.Conv2d}`

        """
        self._torch_to_oflow_cls_map.update(class_proxy_dict)

        debug_message = f"Updated class proxies: {len(class_proxy_dict)}"
        debug_message += f"\n{class_proxy_dict}\n"
        self.logger.debug(debug_message)

    def _transform_entity(self, entity):
        result = self.mocker.mock_entity(entity)
        if result is None:
            RuntimeError(f"Failed to transform entity: {entity}")
        return result

    def get_transformed_entity_name(self, entity):
        return self.mocker.get_mock_entity_name(entity)

    def transform_cls(self, full_cls_name: str):
        """Transform a class name to a mock class ."""
        mock_full_cls_name = self.get_transformed_entity_name(full_cls_name)

        if mock_full_cls_name in self._torch_to_oflow_cls_map:
            use_value = self._torch_to_oflow_cls_map[mock_full_cls_name]
            return use_value

        mock_cls = self._transform_entity(mock_full_cls_name)
        self._torch_to_oflow_cls_map[mock_full_cls_name] = mock_cls
        return mock_cls

    def transform_func(self, func: types.FunctionType):
        # TODO: support transform function cache
        return self._transform_entity(func)

    def transform_package(self, package_name):
        return self._transform_entity(package_name)


debug_mode = os.getenv("ONEDIFF_DEBUG", "0") == "1"
transform_mgr = TransformManager(debug_mode=debug_mode, tmp_dir=None)

try:
    import pydantic

    if pydantic.VERSION < "2.5.2":
        logger.warning(
            f"Pydantic version {pydantic.VERSION} is too low, please upgrade to 2.5.2 or higher."
        )
        from oneflow.mock_torch.mock_utils import MockEnableDisableMixin

        MockEnableDisableMixin.hazard_list.append(
            "huggingface_hub.inference._text_generation"
        )

except ImportError:
    pass
