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
#ifndef ONEFLOW_CORE_PERSISTENCE_STREAM_SCANNER_H_
#define ONEFLOW_CORE_PERSISTENCE_STREAM_SCANNER_H_

#include <vector>
#include <string>
#include "oneflow/core/persistence/binary_in_stream.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class StreamScanner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamScanner);
  virtual ~StreamScanner() {}

  StreamScanner(fs::FileSystem* fs, const std::vector<std::shared_ptr<BinaryInStream>>& streams,
                uint64_t offset);
  bool IsEof() const;
  uint64_t UpdateBuffer(std::vector<char>* buffer);

 protected:
  virtual void AddNForCurFilePos(uint64_t n) = 0;

  std::vector<std::shared_ptr<BinaryInStream>> streams_;
  uint64_t whole_file_size_;
  uint64_t whole_file_pos_;
  int32_t cur_stream_id_;
  int32_t stream_num_;
  uint64_t whole_file_offset_;

 private:
  void AddStream(fs::FileSystem* fs, const std::shared_ptr<BinaryInStream>& stream, int64_t idx);
};

class CyclicStreamScanner final : public StreamScanner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicStreamScanner);
  CyclicStreamScanner(fs::FileSystem* fs,
                      const std::vector<std::shared_ptr<BinaryInStream>>& streams, uint64_t offset)
      : StreamScanner(fs, streams, offset) {}

 protected:
  void AddNForCurFilePos(uint64_t n) override;
};

class AcyclicStreamScanner final : public StreamScanner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AcyclicStreamScanner);
  AcyclicStreamScanner(fs::FileSystem* fs,
                       const std::vector<std::shared_ptr<BinaryInStream>>& streams, uint64_t offset)
      : StreamScanner(fs, streams, offset) {}

 protected:
  void AddNForCurFilePos(uint64_t n) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_STREAM_SCANNER_H_
