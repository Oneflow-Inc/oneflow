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
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/binary_in_stream_with_local_copy.h"
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/job/job_set.pb.h"
#include <cstring>
#include "oneflow/core/common/constant.h"

namespace oneflow {

namespace {

constexpr size_t kDefaultBufferSize = 32 * 1024;  // 32KB

size_t GetBufferSize(int64_t session_id) {
  const auto& io_conf = *Global<const IOConf>::Get(session_id);
  if (io_conf.has_persistence_buf_byte()) {
    const int64_t buffer_size = io_conf.persistence_buf_byte();
    CHECK_GT(buffer_size, 0);
    return buffer_size;
  } else {
    return kDefaultBufferSize;
  }
}

}  // namespace

PersistentInStream::PersistentInStream(fs::FileSystem* fs,
                                       const std::vector<std::string>& file_paths, uint64_t offset,
                                       bool cyclic, bool with_local_copy)
    : PersistentInStream(kInvalidSessionId, fs, file_paths, offset, cyclic, with_local_copy) {}

PersistentInStream::PersistentInStream(int64_t session_id, fs::FileSystem* fs,
                                       const std::vector<std::string>& file_paths, uint64_t offset,
                                       bool cyclic, bool with_local_copy) {
  if (with_local_copy) { CHECK_EQ(offset, 0); }
  std::vector<std::shared_ptr<BinaryInStream>> streams;
  for (auto& file_path : file_paths) {
    if (with_local_copy) {
      streams.emplace_back(new BinaryInStreamWithLocalCopy(fs, file_path));
    } else {
      streams.emplace_back(new BinaryInStreamWithoutLocalCopy(fs, file_path));
    }
  }
  if (cyclic) {
    stream_scanner_.reset(new CyclicStreamScanner(fs, streams, offset));
  } else {
    stream_scanner_.reset(new AcyclicStreamScanner(fs, streams, offset));
  }
  buffer_.resize(GetBufferSize(session_id) + 1);
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data();
  *cur_buf_end_ = '\0';
}

PersistentInStream::PersistentInStream(fs::FileSystem* fs,
                                       const std::vector<std::string>& file_paths, bool cyclic,
                                       bool with_local_copy)
    : PersistentInStream(fs, file_paths, 0, cyclic, with_local_copy) {}

PersistentInStream::PersistentInStream(fs::FileSystem* fs, const std::string& file_path,
                                       uint64_t offset, bool cyclic, bool with_local_copy)
    : PersistentInStream(fs, std::vector<std::string>({file_path}), offset, cyclic,
                         with_local_copy) {}

PersistentInStream::PersistentInStream(fs::FileSystem* fs, const std::string& file_path,
                                       uint64_t offset)
    : PersistentInStream(fs, file_path, offset, false, false) {}

PersistentInStream::PersistentInStream(fs::FileSystem* fs, const std::string& file_path)
    : PersistentInStream(fs, file_path, 0, false, false) {}

PersistentInStream::PersistentInStream(int64_t session_id, fs::FileSystem* fs,
                                       const std::string& file_path)
    : PersistentInStream(session_id, fs, std::vector<std::string>({file_path}), 0, false, false) {}

int32_t PersistentInStream::ReadLine(std::string* l) {
  if (IsEof()) { return -1; }
  l->clear();
  while (*cur_buf_begin_ != '\n') {
    if (cur_buf_begin_ == cur_buf_end_) {
      UpdateBuffer();
      if (cur_buf_begin_ == cur_buf_end_) {
        return 0;
      } else {
        continue;
      }
    }
    l->push_back(*cur_buf_begin_++);
  }
  ++cur_buf_begin_;
  return 0;
}

int32_t PersistentInStream::ReadFully(char* s, size_t n) {
  if (IsEof()) { return -1; }
  while (n) {
    if (cur_buf_begin_ == cur_buf_end_) { UpdateBuffer(); }
    CHECK_LT(cur_buf_begin_, cur_buf_end_);
    int64_t copy_size = std::min<int64_t>(cur_buf_end_ - cur_buf_begin_, n);
    std::memcpy(s, cur_buf_begin_, static_cast<size_t>(copy_size));
    s += copy_size;
    cur_buf_begin_ += copy_size;
    n -= copy_size;
  }
  return 0;
}

void PersistentInStream::UpdateBuffer() {
  CHECK_EQ(cur_buf_begin_, cur_buf_end_);
  uint64_t n = stream_scanner_->UpdateBuffer(&buffer_);
  cur_buf_begin_ = buffer_.data();
  cur_buf_end_ = buffer_.data() + n;
  *cur_buf_end_ = '\0';
}

bool PersistentInStream::IsEof() const {
  return cur_buf_begin_ == cur_buf_end_ && stream_scanner_->IsEof();
}
}  // namespace oneflow
