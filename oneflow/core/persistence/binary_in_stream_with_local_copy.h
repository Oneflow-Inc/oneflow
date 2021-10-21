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
#ifndef ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_WITH_LOCAL_COPY_H_
#define ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_WITH_LOCAL_COPY_H_

#include "oneflow/core/persistence/binary_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class BinaryInStreamWithLocalCopy final : public BinaryInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryInStreamWithLocalCopy);
  BinaryInStreamWithLocalCopy() = delete;
  ~BinaryInStreamWithLocalCopy() = default;

  BinaryInStreamWithLocalCopy(fs::FileSystem* fs, const std::string& file_path);

  int32_t Read(char* s, size_t n) override { return (this->*read_mthd_)(s, n); }

  uint64_t file_size() const override { return in_stream_->file_size(); }
  uint64_t cur_file_pos() const override { return in_stream_->cur_file_pos(); }
  void set_cur_file_pos(uint64_t val) override { in_stream_->set_cur_file_pos(val); }
  bool IsEof() const override { return in_stream_->IsEof(); }

 private:
  int32_t ReadAndWriteToLocal(char* s, size_t n);
  int32_t ReadFromLocal(char* s, size_t n) { return in_stream_->Read(s, n); }

  bool Restart();
  void CopyToLocalFinish();

  bool once_read_;
  std::unique_ptr<BinaryInStream> in_stream_;
  std::string local_copy_path_;
  std::unique_ptr<PersistentOutStream> out_stream_;
  int32_t (BinaryInStreamWithLocalCopy::*read_mthd_)(char*, size_t);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_WITH_LOCAL_COPY_H_
