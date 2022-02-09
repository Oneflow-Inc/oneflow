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
from google.protobuf import text_format

import oneflow._oneflow_internal
import oneflow.framework.scope_util as scope_util


def MakeScopeSymbol(job_conf, parallel_conf, is_mirrored):
    parallel_hierarchy = None
    if parallel_conf.has_hierarchy():
        parallel_hierarchy = oneflow._oneflow_internal.Size(
            tuple(parallel_conf.hierarchy().dim())
        )
    return scope_util.MakeInitialScope(
        job_conf,
        parallel_conf.device_tag(),
        list(parallel_conf.device_name()),
        parallel_hierarchy,
        is_mirrored,
    ).symbol_id
