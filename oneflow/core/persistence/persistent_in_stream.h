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
#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/stream_scanner.h"

namespace oneflow {

class PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentInStream);
  virtual ~PersistentInStream() {}
  PersistentInStream(fs::FileSystem* fs, const std::vector<std::string>& file_paths,
                     uint64_t offset, bool cyclic, bool with_local_copy);
  PersistentInStream(fs::FileSystem* fs, const std::vector<std::string>& file_paths, bool cyclic,
                     bool with_local_copy);
  PersistentInStream(fs::FileSystem* fs, const std::string& file_path, uint64_t offset, bool cyclic,
                     bool with_local_copy);
  PersistentInStream(fs::FileSystem* fs, const std::string& file_path, uint64_t offset);
  PersistentInStream(fs::FileSystem* fs, const std::string& file_path);
  PersistentInStream(int64_t session_id, fs::FileSystem* fs, const std::string& file_path);

  PersistentInStream(int64_t session_id, fs::FileSystem* fs,
                     const std::vector<std::string>& file_paths, uint64_t offset, bool cyclic,
                     bool with_local_copy);

  // 0: success
  // -1: eof
  int32_t ReadLine(std::string* l);
  int32_t ReadFully(char* s, size_t n);

 private:
  bool IsEof() const;
  void UpdateBuffer();

  std::unique_ptr<StreamScanner> stream_scanner_;

  std::vector<char> buffer_;
  char* cur_buf_begin_;
  char* cur_buf_end_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
