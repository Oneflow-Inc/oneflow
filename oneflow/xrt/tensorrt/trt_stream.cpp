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
#include "oneflow/xrt/tensorrt/trt_stream.h"

#include <mutex>
#include "oneflow/core/common/hash_container.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

namespace {

struct StreamTableImpl {
  std::mutex mutex;
  HashMap<uint64_t, ep::Stream*> streams;
};

static StreamTableImpl st;

}  // namespace

void RecordStream(uint64_t stream_id, ep::Stream* stream) {
  std::unique_lock<std::mutex> lock(st.mutex);
  const auto& it = st.streams.find(stream_id);
  if (it != st.streams.end()) {
    CHECK_EQ(it->second, stream);
  } else {
    st.streams.emplace(stream_id, stream).first;
  }
}

ep::Stream* LookupStream(uint64_t stream_id) {
  std::unique_lock<std::mutex> lock(st.mutex);
  const auto& it = st.streams.find(stream_id);
  CHECK(it != st.streams.end());
  return it->second;
}

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow
