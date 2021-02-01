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
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate


@oneflow_export("stage")
class Stage(object):
    def __init__(
        self,
        placement,
        stage_load=1,
        stage_placement_id=None,
        stage_weight_buffer_size=None,
    ):
        self.placement_ = placement
        self.stage_load_ = stage_load
        self.stage_placement_id_ = stage_placement_id
        self.stage_weight_buffer_size_ = stage_weight_buffer_size

    @property
    def placement(self):
        return self.placement_

    @property
    def stage_load(self):
        return self.stage_load_

    @property
    def stage_placement_id(self):
        return self.stage_placement_id_

    @property
    def stage_weight_buffer_size(self):
        return self.stage_weight_buffer_size_
