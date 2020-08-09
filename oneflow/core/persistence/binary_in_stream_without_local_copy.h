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
#ifndef ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_WITHOUT_LOCAL_COPY_H_
#define ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_WITHOUT_LOCAL_COPY_H_

#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/binary_in_stream.h"

namespace oneflow {

class BinaryInStreamWithoutLocalCopy final : public BinaryInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryInStreamWithoutLocalCopy);
  BinaryInStreamWithoutLocalCopy() = delete;
  virtual ~BinaryInStreamWithoutLocalCopy() = default;

  BinaryInStreamWithoutLocalCopy(fs::FileSystem*, const std::string& file_path);
  int32_t Read(char* s, size_t n) override;

  uint64_t file_size() const override { return file_size_; }
  uint64_t cur_file_pos() const override { return cur_file_pos_; }
  void set_cur_file_pos(uint64_t val) override { cur_file_pos_ = val; }
  bool IsEof() const override { return cur_file_pos_ == file_size_; }

 private:
  std::unique_ptr<fs::RandomAccessFile> file_;
  uint64_t file_size_;
  uint64_t cur_file_pos_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_WITHOUT_LOCAL_COPY_H_
