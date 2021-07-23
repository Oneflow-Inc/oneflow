/*
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
*/
#include "oneflow/core/vm/stream_desc.msg.h"

namespace oneflow {
namespace vm {

void StreamDesc::__Init__(const StreamTypeId& stream_type_id, int32_t num_machines,
                          int32_t num_streams_per_machine, int32_t num_streams_per_thread) {
  mutable_stream_type_id()->CopyFrom(stream_type_id);
  set_num_machines(num_machines);
  set_num_streams_per_machine(num_streams_per_machine);
  set_num_streams_per_thread(num_streams_per_thread);
}

int32_t StreamDesc::num_threads() const {
  int32_t num_devices = num_machines() * num_streams_per_machine();
  CHECK_EQ(num_devices % num_streams_per_thread(), 0);
  return num_devices / num_streams_per_thread();
}

}  // namespace vm
}  // namespace oneflow
