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
#ifndef ONEFLOW_USER_DATA_MMAP_FILE_H_
#define ONEFLOW_USER_DATA_MMAP_FILE_H_

#include <string>
#include <stddef.h>

namespace oneflow {

namespace data {

class MMapFile final {
 public:
  MMapFile(const std::string& file_path);
  ~MMapFile();

  void read(void* buf, size_t offset, size_t length) const;
  void read(void* buf, size_t length);

 private:
  size_t size_;
  size_t offset_;
  char* mapped_;
};

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_MMAP_FILE_H_
