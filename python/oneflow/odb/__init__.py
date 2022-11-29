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

import oneflow._oneflow_internal

b_normal = oneflow._oneflow_internal.odb.SetNormalBreakpoint
b_scheduler = oneflow._oneflow_internal.odb.SetSchedulerBreakpoint
b_worker = oneflow._oneflow_internal.odb.SetWorkerBreakpoint
b = oneflow._oneflow_internal.odb.SetAllBreakpoints

d_normal = oneflow._oneflow_internal.odb.ClearNormalBreakpoint
d_scheduler = oneflow._oneflow_internal.odb.ClearSchedulerBreakpoint
d_worker = oneflow._oneflow_internal.odb.ClearWorkerBreakpoint
d = oneflow._oneflow_internal.odb.ClearAllBreakpoints

stop_vm_scheduler = oneflow._oneflow_internal.odb.StopVMScheduler
restart_vm_scheduler = oneflow._oneflow_internal.odb.RestartVMScheduler

stop_vm_worker = oneflow._oneflow_internal.odb.StopVMWorker
restart_vm_worker = oneflow._oneflow_internal.odb.RestartVMWorker
