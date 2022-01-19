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
#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_OUT_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_OUT_STREAM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class PersistentOutStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentOutStream);
  PersistentOutStream() = delete;
  ~PersistentOutStream();

  PersistentOutStream(fs::FileSystem*, const std::string& file_path);

  // Write block of data
  // Inserts the first n characters of the array pointed by s into the stream.
  PersistentOutStream& Write(const char* s, size_t n);

  void Flush();

 private:
  std::unique_ptr<fs::WritableFile> file_;
};

template<typename T>
typename std::enable_if<std::is_fundamental<T>::value, PersistentOutStream&>::type operator<<(
    PersistentOutStream& out_stream, const T& x) {
  const char* x_ptr = reinterpret_cast<const char*>(&x);
  size_t n = sizeof(x);
  out_stream.Write(x_ptr, n);
  return out_stream;
}

inline PersistentOutStream& operator<<(PersistentOutStream& out_stream, const std::string& s) {
  out_stream.Write(s.c_str(), s.size());
  return out_stream;
}

template<size_t n>
PersistentOutStream& operator<<(PersistentOutStream& out_stream, const char (&s)[n]) {
  out_stream.Write(s, strlen(s));
  return out_stream;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCY_PERSISTENT_OUT_STREAM_H
