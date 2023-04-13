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

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <sys/mman.h>
#include <libgen.h>
#include <dirent.h>
#include <sys/file.h>

namespace oneflow {

namespace embedding {

class PosixFile final {
 public:
  PosixFile() : fd_(-1), size_(0) {}
  PosixFile(const std::string& pathname, int flags, mode_t mode)
      : PosixFile(pathname.c_str(), flags, mode) {}
  PosixFile(const char* pathname, int flags, mode_t mode) : PosixFile() {
    fd_ = open(pathname, flags, mode);
    PCHECK(fd_ != -1);
    struct stat sb {};
    PCHECK(fstat(fd_, &sb) == 0);
    size_ = sb.st_size;
  }
  PosixFile(PosixFile&& other) noexcept : PosixFile() { *this = std::move(other); }
  PosixFile(const PosixFile&) = delete;
  ~PosixFile() { Close(); }

  PosixFile& operator=(PosixFile&& other) noexcept {
    this->Close();
    fd_ = other.fd_;
    other.fd_ = -1;
    size_ = other.size_;
    other.size_ = 0;
    return *this;
  }
  PosixFile& operator=(const PosixFile&) = delete;

  int fd() { return fd_; }

  bool IsOpen() { return fd_ != -1; }

  void Close() {
    if (IsOpen()) {
      PCHECK(close(fd_) == 0);
      fd_ = -1;
    }
  }

  size_t Size() { return size_; }

  void Truncate(size_t new_size) {
    CHECK(IsOpen());
    if (new_size == size_) { return; }
    PCHECK(ftruncate(fd_, new_size) == 0);
    size_ = new_size;
  }

  static bool FileExists(const std::string& pathname) {
    return access(pathname.c_str(), F_OK) == 0;
  }

  static std::string JoinPath(const std::string& a, const std::string& b) { return a + "/" + b; }

  static void RecursiveCreateDirectory(const std::string& pathname, mode_t mode) {
    while (true) {
      struct stat sb {};
      if (stat(pathname.c_str(), &sb) == 0) {
        CHECK(S_ISDIR(sb.st_mode)) << "Could not create directory: '" << pathname
                                   << "' already exists and is not a directory.";
        return;
      } else {
        PCHECK(errno == ENOENT) << "Could not create directory '" << pathname << "'.";
        if (lstat(pathname.c_str(), &sb) == 0) {
          LOG(FATAL) << "Could not create directory: '" << pathname << "' is a broken link.";
        } else {
          PCHECK(errno == ENOENT) << "Could not create directory '" << pathname << "'.";
        }
        std::vector<char> dirname_input(pathname.size() + 1);
        std::memcpy(dirname_input.data(), pathname.c_str(), pathname.size() + 1);
        const std::string parent = dirname(dirname_input.data());
        RecursiveCreateDirectory(parent, mode);
        if (mkdir(pathname.c_str(), mode) == 0) {
          return;
        } else {
          PCHECK(errno == EEXIST) << "Could not create directory '" << pathname << "'.";
        }
      }
    }
  }

  static void RecursiveDelete(const std::string& pathname) {
    struct stat sb {};
    if (stat(pathname.c_str(), &sb) == 0) {
      if (S_ISDIR(sb.st_mode)) {
        DIR* dir = opendir(pathname.c_str());
        PCHECK(dir != nullptr);
        struct dirent* ent = nullptr;
        while ((ent = readdir(dir)) != nullptr) {
          if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) { continue; }
          RecursiveDelete(pathname + "/" + ent->d_name);
        }
        PCHECK(closedir(dir) == 0);
        PCHECK(rmdir(pathname.c_str()) == 0);
      } else {
        PCHECK(unlink(pathname.c_str()) == 0);
      }
    } else {
      PCHECK(errno == ENOENT);
    }
  }

 private:
  int fd_;
  size_t size_;
};

class PosixMappedFile final {
 public:
  PosixMappedFile() : file_(), ptr_(nullptr) {}
  PosixMappedFile(PosixFile&& file, size_t size, int prot, int flags)
      : file_(std::move(file)), ptr_(nullptr) {
    CHECK_NE(file_.fd(), -1);
    void* ptr = mmap(nullptr, size, prot, flags, file_.fd(), 0);
    PCHECK(ptr != MAP_FAILED);
    ptr_ = ptr;
  }
  PosixMappedFile(PosixFile&& file, size_t size, int prot)
      : PosixMappedFile(std::move(file), size, prot, MAP_SHARED) {}
  PosixMappedFile(PosixMappedFile&& other) noexcept : PosixMappedFile() {
    *this = std::move(other);
  }
  PosixMappedFile(const PosixMappedFile&) = delete;
  ~PosixMappedFile() { Unmap(); }

  PosixMappedFile& operator=(PosixMappedFile&& other) noexcept {
    Unmap();
    this->file_ = std::move(other.file_);
    this->ptr_ = other.ptr_;
    other.ptr_ = nullptr;
    return *this;
  }
  PosixMappedFile& operator=(const PosixMappedFile&) = delete;

  void* ptr() { return ptr_; }

  PosixFile& file() { return file_; }

 private:
  void Unmap() {
    if (ptr_ != nullptr) { PCHECK(munmap(ptr_, file_.Size()) == 0); }
  }
  PosixFile file_;
  void* ptr_;
};

class PosixFileLockGuard final {
 public:
  OF_DISALLOW_COPY(PosixFileLockGuard);
  explicit PosixFileLockGuard() : file_() {}
  explicit PosixFileLockGuard(PosixFile&& file) : file_(std::move(file)) {
    CHECK_NE(file_.fd(), -1);
    Lock();
  }
  PosixFileLockGuard(PosixFileLockGuard&& other) noexcept { *this = std::move(other); }
  PosixFileLockGuard& operator=(PosixFileLockGuard&& other) noexcept {
    Unlock();
    file_ = std::move(other.file_);
    return *this;
  }
  ~PosixFileLockGuard() { Unlock(); }

 private:
  void Lock() {
    if (file_.fd() != -1) {
      struct flock f {};
      f.l_type = F_WRLCK;
      f.l_whence = SEEK_SET;
      f.l_start = 0;
      f.l_len = 0;
      PCHECK(fcntl(file_.fd(), F_SETLK, &f) == 0);
    }
  }
  void Unlock() {
    if (file_.fd() != -1) {
      struct flock f {};
      f.l_type = F_UNLCK;
      f.l_whence = SEEK_SET;
      f.l_start = 0;
      f.l_len = 0;
      PCHECK(fcntl(file_.fd(), F_SETLK, &f) == 0);
    }
  }

  PosixFile file_;
};

}  // namespace embedding

}  // namespace oneflow

#endif  // __linux__

#endif  // ONEFLOW_CORE_EMBEDDING_POSIX_FILE_H_
