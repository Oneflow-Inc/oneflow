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
#ifndef ONEFLOW_CORE_EMBEDDING_FILE_HANDLE_H_
#define ONEFLOW_CORE_EMBEDDING_FILE_HANDLE_H_

#include "oneflow/core/common/util.h"
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <sys/mman.h>

namespace oneflow {

namespace embedding {

inline bool FileExists(const char* name) { return access(name, F_OK) == 0; }

class FileHandle final {
 public:
  OF_DISALLOW_COPY(FileHandle);
  FileHandle() : fd_(-1), size_(0) {}
  FileHandle(const char* pathname, int flags, mode_t mode) : FileHandle() {
    fd_ = open(pathname, flags, mode);
    PCHECK(fd_ != -1);
    struct stat sb {};
    PCHECK(fstat(fd_, &sb) == 0);
    size_ = sb.st_size;
  }
  FileHandle(FileHandle&& other) noexcept : FileHandle() { *this = std::move(other); }
  FileHandle& operator=(FileHandle&& other) noexcept {
    this->Close();
    fd_ = other.fd_;
    other.fd_ = -1;
    size_ = other.size_;
    other.size_ = 0;
    return *this;
  }
  ~FileHandle() { Close(); }

  int fd() { return fd_; }

  void Close() {
    if (fd_ != -1) {
      PCHECK(close(fd_) == 0);
      fd_ = -1;
    }
  }

  size_t Size() { return size_; }

  void Truncate(size_t new_size) {
    CHECK_NE(fd_, -1);
    if (new_size == size_) { return; }
    PCHECK(ftruncate(fd_, new_size) == 0);
    size_ = new_size;
  }

 private:
  int fd_;
  size_t size_;
};

class MappedFileHandle final {
 public:
  OF_DISALLOW_COPY(MappedFileHandle);
  MappedFileHandle() : file_(), ptr_(nullptr) {}
  explicit MappedFileHandle(FileHandle&& file, int prot) : file_(std::move(file)), ptr_(nullptr) {
    CHECK_NE(file_.fd(), -1);
    void* ptr = mmap(nullptr, file_.Size(), prot, MAP_SHARED, file_.fd(), 0);
    CHECK_NE(ptr, MAP_FAILED);
    ptr_ = ptr;
  }
  MappedFileHandle(MappedFileHandle&& other) noexcept : MappedFileHandle() {
    *this = std::move(other);
  }
  MappedFileHandle& operator=(MappedFileHandle&& other) noexcept {
    Unmap();
    this->file_ = std::move(other.file_);
    this->ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }
  ~MappedFileHandle() { Unmap(); }

  void* ptr() { return ptr_; }

 private:
  void Unmap() {
    if (ptr_ != nullptr) { PCHECK(munmap(ptr_, file_.Size()) == 0); }
  }
  FileHandle file_;
  void* ptr_;
};

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_FILE_HANDLE_H_
