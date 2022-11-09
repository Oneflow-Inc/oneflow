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
#include "oneflow/core/persistence/binary_in_stream_without_local_copy.h"
#include "oneflow/core/job/job_desc.h"
#include <cstring>

namespace oneflow {

int32_t BinaryInStreamWithoutLocalCopy::Read(char* s, size_t n) {
  if (IsEof()) return -1;
  CHECK_LE(cur_file_pos_ + n, file_size_);
  file_->Read(cur_file_pos_, n, s);
  cur_file_pos_ += n;
  return 0;
}

BinaryInStreamWithoutLocalCopy::BinaryInStreamWithoutLocalCopy(fs::FileSystem* fs,
                                                               const std::string& file_path)
    : cur_file_pos_(0) {
  fs->NewRandomAccessFile(file_path, &file_);
  file_size_ = fs->GetFileSize(file_path);
}

}  // namespace oneflow
