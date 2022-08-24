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
#include "oneflow/core/framework/stream_mgr.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

Maybe<Symbol<Stream>> StreamMgr::AddStreamSymbol(
    const Stream& stream,
    const std::function<Maybe<Symbol<Stream>>(size_t unique_stream_id)>& CreateStreamSymbol) {
  Symbol<Stream> stream_symbol;
  std::unique_lock<std::mutex> lock(mutex_);
  if (stream2unique_stream_id_.count(stream) > 0) {
    size_t unique_stream_id = stream2unique_stream_id_[stream];
    auto existed_stream_symbol = JUST(VectorAt(unique_stream_id2stream_symbol_, unique_stream_id));
    stream_symbol = JUST(CreateStreamSymbol(unique_stream_id));
    CHECK_OR_RETURN(existed_stream_symbol == stream_symbol)
        << "the result of current called CreateStreamSymbol is not the result of last called "
           "CreateStreamSymbol";
  } else {
    size_t unique_stream_id = unique_stream_id2stream_symbol_.size();
    stream2unique_stream_id_[stream] = unique_stream_id;
    stream_symbol = JUST(CreateStreamSymbol(unique_stream_id));
    unique_stream_id2stream_symbol_.push_back(stream_symbol);
    CHECK_OR_RETURN(unique_stream_id2stream_symbol_[unique_stream_id] == stream)
        << "the result of CreateStreamSymbol is no the symbol of `stream`";
    CHECK_EQ_OR_RETURN(unique_stream_id2stream_symbol_[unique_stream_id]->unique_stream_id(),
                       unique_stream_id)
        << "unique_stream_id is wrongly initialized";
  }
  return stream_symbol;
}

size_t StreamMgr::UniqueStreamSize() const {
  std::unique_lock<std::mutex> lock(mutex_);
  return unique_stream_id2stream_symbol_.size();
}

Maybe<Symbol<Stream>> StreamMgr::GetStreamSymbol(size_t unique_stream_id) const {
  std::unique_lock<std::mutex> lock(mutex_);
  return JUST(VectorAt(unique_stream_id2stream_symbol_, unique_stream_id));
}

COMMAND(Singleton<StreamMgr>::SetAllocated(new StreamMgr()));

}  // namespace oneflow
