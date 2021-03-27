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
#include "oneflow/user/data/mmap_file.h"
#include "oneflow/core/common/util.h"
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>

namespace oneflow {

namespace data {

MMapFile::MMapFile(const std::string& filepath) : offset_(0) {
#ifdef __linux__
  int fd = open(filepath.c_str(), O_RDONLY);
  CHECK(fd != -1) << "open " << filepath << " failed: " << strerror(errno);

  struct stat s;
  CHECK(fstat(fd, &s) != -1) << "stat " << filepath << " failed: " << strerror(errno);
  size_ = s.st_size;

  void* ptr = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  CHECK(ptr != MAP_FAILED) << "mmap " << filepath << " failed: " << strerror(errno);

  close(fd);

  mapped_ = static_cast<char*>(ptr);
#else
  UNIMPLEMENTED();
#endif
}

MMapFile::~MMapFile() {
#ifdef __linux__
  CHECK(munmap(mapped_, size_) == 0) << "munmap failed";
#else
  UNIMPLEMENTED();
#endif
}

void MMapFile::read(void* buf, size_t offset, size_t length) const {
  memcpy(buf, mapped_ + offset, length);
}

void MMapFile::read(void* buf, size_t length) {
  read(buf, offset_, length);
  offset_ += length;
}

}  // namespace data

}  // namespace oneflow
