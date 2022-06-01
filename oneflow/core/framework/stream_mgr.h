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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_MGR_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_MGR_H_

#include <mutex>
#include <functional>
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/stream.h"

namespace oneflow {

class StreamMgr final {
 public:
  StreamMgr() = default;
  ~StreamMgr() = default;

  Maybe<Symbol<Stream>> AddStreamSymbol(
      const Stream& stream,
      const std::function<Maybe<Symbol<Stream>>(size_t unique_stream_id)>& CreateStreamSymbol);

  size_t UniqueStreamSize() const;

  Maybe<Symbol<Stream>> GetStreamSymbol(size_t unique_stream_id) const;

 private:
  mutable std::mutex mutex_;
  std::vector<Symbol<Stream>> unique_stream_id2stream_symbol_;
  std::unordered_map<Stream, size_t> stream2unique_stream_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_MGR_H_
