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
#ifndef ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class BinaryInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryInStream);
  virtual ~BinaryInStream() = default;

  // 0: success
  // -1: eof
  virtual int32_t Read(char* s, size_t n) = 0;

  virtual uint64_t file_size() const = 0;
  virtual uint64_t cur_file_pos() const = 0;
  virtual void set_cur_file_pos(uint64_t val) = 0;
  virtual bool IsEof() const = 0;

 protected:
  BinaryInStream() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_BINARY_IN_STREAM_H_
