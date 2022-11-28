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
#ifndef ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class LogStreamDestination final {
 public:
  LogStreamDestination(fs::FileSystem* file_system, const std::string& base_dir)
      : file_system_(file_system), base_dir_(base_dir) {}
  ~LogStreamDestination() = default;
  fs::FileSystem* mut_file_system() const { return file_system_; };
  const std::string& base_dir() const { return base_dir_; };

 private:
  fs::FileSystem* file_system_;
  std::string base_dir_;
};

class TeePersistentLogStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TeePersistentLogStream);
  ~TeePersistentLogStream();

  void Write(const char* s, size_t n);
  void Write(const std::string& str);
  void Write(const PbMessage& proto);

  static std::unique_ptr<TeePersistentLogStream> Create(const std::string& path);
  void Flush();

 private:
  explicit TeePersistentLogStream(const std::string& path);
  std::vector<LogStreamDestination> destinations_;
  std::vector<std::unique_ptr<PersistentOutStream>> branches_;
};

inline TeePersistentLogStream& operator<<(TeePersistentLogStream& log_stream,
                                          const std::string& s) {
  log_stream.Write(s.c_str(), s.size());
  return log_stream;
}

inline std::unique_ptr<TeePersistentLogStream>& operator<<(
    std::unique_ptr<TeePersistentLogStream>& log_stream, const std::string& s) {
  log_stream->Write(s.c_str(), s.size());
  return log_stream;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_
