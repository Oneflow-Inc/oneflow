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
#include "oneflow/core/vm/stream_runtime_desc.msg.h"

namespace oneflow {
namespace vm {

void StreamRtDesc::__Init__(StreamDesc* stream_desc) {
  const StreamTypeId& stream_type_id = stream_desc->stream_type_id();
  reset_stream_desc(stream_desc);
  mutable_stream_type_id()->CopyFrom(stream_type_id);
}

const StreamType& StreamRtDesc::stream_type() const { return stream_type_id().stream_type(); }

}  // namespace vm
}  // namespace oneflow
