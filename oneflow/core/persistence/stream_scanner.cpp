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
#include "oneflow/core/persistence/stream_scanner.h"
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/persistence/binary_in_stream_with_local_copy.h"

namespace oneflow {

StreamScanner::StreamScanner(fs::FileSystem* fs,
                             const std::vector<std::shared_ptr<BinaryInStream>>& streams,
                             uint64_t offset)
    : whole_file_offset_(offset) {
  stream_num_ = streams.size();
  whole_file_size_ = 0;
  int64_t idx = 0;
  for (auto& stream : streams) {
    AddStream(fs, stream, idx);
    ++idx;
  }
  CHECK_LE(whole_file_offset_, whole_file_size_);
  whole_file_pos_ = whole_file_offset_;
}

void StreamScanner::AddStream(fs::FileSystem* fs, const std::shared_ptr<BinaryInStream>& stream,
                              int64_t idx) {
  uint64_t cur_file_size = stream->file_size();
  if (whole_file_offset_ < whole_file_size_) {
    stream->set_cur_file_pos(0);
  } else if (whole_file_size_ <= whole_file_offset_
             && whole_file_offset_ < whole_file_size_ + cur_file_size) {
    stream->set_cur_file_pos(whole_file_offset_ - whole_file_size_);
    cur_stream_id_ = idx;
  } else if (whole_file_offset_ >= whole_file_size_ + cur_file_size) {
    stream->set_cur_file_pos(0);  // works for both cyclic and acyclic cases
  }

  streams_.emplace_back(stream);
  whole_file_size_ += cur_file_size;
}

bool StreamScanner::IsEof() const { return whole_file_pos_ == whole_file_size_; }

uint64_t StreamScanner::UpdateBuffer(std::vector<char>* buffer) {
  if (cur_stream_id_ == stream_num_) return 0;
  uint64_t n =
      std::min(static_cast<uint64_t>(buffer->size() - 1),
               streams_[cur_stream_id_]->file_size() - streams_[cur_stream_id_]->cur_file_pos());
  if (n == 0) { return 0; }
  streams_[cur_stream_id_]->Read(buffer->data(), n);
  AddNForCurFilePos(n);
  return n;
}

void AcyclicStreamScanner::AddNForCurFilePos(uint64_t n) {
  whole_file_pos_ += n;
  if (streams_[cur_stream_id_]->IsEof()) { ++cur_stream_id_; }
}

void CyclicStreamScanner::AddNForCurFilePos(uint64_t n) {
  whole_file_pos_ = (whole_file_pos_ + n) % whole_file_size_;
  if (streams_[cur_stream_id_]->IsEof()) {
    streams_[cur_stream_id_]->set_cur_file_pos(0);
    ++cur_stream_id_;
    if (cur_stream_id_ == stream_num_) {
      CHECK_EQ(whole_file_pos_, 0);
      cur_stream_id_ = 0;
    }
  }
}

}  // namespace oneflow
