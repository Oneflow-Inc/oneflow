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
#ifndef ONEFLOW_CORE_EMBEDDING_POSIX_FILE_H_
#define ONEFLOW_CORE_EMBEDDING_POSIX_FILE_H_

#ifdef __linux__

#include "oneflow/core/common/util.h"
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <sys/mman.h>
#include <libgen.h>

namespace oneflow {

namespace embedding {

class PosixFile final {
 public:
  OF_DISALLOW_COPY(PosixFile);
  PosixFile() : fd_(-1), size_(0) {}
  PosixFile(const char* pathname, int flags, mode_t mode) : PosixFile() {
    fd_ = open(pathname, flags, mode);
    PCHECK(fd_ != -1);
    struct stat sb {};
    PCHECK(fstat(fd_, &sb) == 0);
    size_ = sb.st_size;
  }
  PosixFile(PosixFile&& other) noexcept : PosixFile() { *this = std::move(other); }
  PosixFile& operator=(PosixFile&& other) noexcept {
    this->Close();
    fd_ = other.fd_;
    other.fd_ = -1;
    size_ = other.size_;
    other.size_ = 0;
    return *this;
  }
  ~PosixFile() { Close(); }

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

  static bool FileExists(const char* name) { return access(name, F_OK) == 0; }

  static void CreateDirectoryIfNotExists(const std::string& pathname, mode_t mode) {
    while (true) {
      struct stat sb {};
      if (stat(pathname.c_str(), &sb) == 0) {
        CHECK(S_ISDIR(sb.st_mode)) << "'" << pathname << "' already exists and is not a directory.";
        return;
      } else {
        PCHECK(errno == ENOENT);
        std::vector<char> dirname_input(pathname.size() + 1);
        std::memcpy(dirname_input.data(), pathname.c_str(), pathname.size() + 1);
        const std::string parent = dirname(dirname_input.data());
        CreateDirectoryIfNotExists(parent, mode);
        if (mkdir(pathname.c_str(), mode) == 0) {
          return;
        } else {
          PCHECK(errno == EEXIST);
        }
      }
    }
  }

 private:
  int fd_;
  size_t size_;
};

class PosixMappedFile final {
 public:
  OF_DISALLOW_COPY(PosixMappedFile);
  PosixMappedFile() : file_(), ptr_(nullptr) {}
  PosixMappedFile(PosixFile&& file, size_t size, int prot) : file_(std::move(file)), ptr_(nullptr) {
    CHECK_NE(file_.fd(), -1);
    void* ptr = mmap(nullptr, size, prot, MAP_SHARED, file_.fd(), 0);
    CHECK_NE(ptr, MAP_FAILED);
    ptr_ = ptr;
  }
  PosixMappedFile(PosixMappedFile&& other) noexcept : PosixMappedFile() {
    *this = std::move(other);
  }
  PosixMappedFile& operator=(PosixMappedFile&& other) noexcept {
    Unmap();
    this->file_ = std::move(other.file_);
    this->ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }
  ~PosixMappedFile() { Unmap(); }

  void* ptr() { return ptr_; }

  PosixFile& file() { return file_; }

 private:
  void Unmap() {
    if (ptr_ != nullptr) { PCHECK(munmap(ptr_, file_.Size()) == 0); }
  }
  PosixFile file_;
  void* ptr_;
};

}  // namespace embedding

}  // namespace oneflow

#endif  // __linux__

#endif  // ONEFLOW_CORE_EMBEDDING_POSIX_FILE_H_
