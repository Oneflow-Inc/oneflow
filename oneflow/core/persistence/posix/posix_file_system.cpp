#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "oneflow/core/persistence/posix/posix_file_system.h"

namespace oneflow {

namespace fs {

class PosixRandomAccessFile : public RandomAccessFile {
 private:
  std::string fname_;
  int fd_;

 public:
  PosixRandomAccessFile(const std::string& fname, int fd)
      : fname_(fname), fd_(fd) {}
  ~PosixRandomAccessFile() override { close(fd_); }

  Status Read(uint64_t offset, size_t n, char* result) const override {
    char* dst = result;
    while (n > 0) {
      ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        return Status::OUT_OF_RANGE;
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        return ErrnoToStatus(errno);
      }
    }
    return Status::OK;
  }
};

class PosixWritableFile : public WritableFile {
 private:
  std::string fname_;
  FILE* file_;

 public:
  PosixWritableFile(const std::string& fname, FILE* file)
      : fname_(fname), file_(file) {}

  ~PosixWritableFile() override {
    if (file_ != nullptr) { fclose(file_); }
  }

  Status Append(const char* data, size_t n) override {
    size_t r = fwrite(data, sizeof(char), n, file_);
    if (r != n) { return ErrnoToStatus(errno); }
    return Status::OK;
  }

  Status Close() override {
    Status s = Status::OK;
    if (fclose(file_) != 0) { s = ErrnoToStatus(errno); }
    file_ = nullptr;
    return s;
  }

  Status Flush() override {
    if (fflush(file_) != 0) { return ErrnoToStatus(errno); }
    return Status::OK;
  }

  Status Sync() {
    Status s = Status::OK;
    if (fflush(file_) != 0) { s = ErrnoToStatus(errno); }
    return s;
  }
};

Status PosixFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  std::string translated_fname = TranslateName(fname);
  Status s = Status::OK;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = ErrnoToStatus(errno);
  } else {
    result->reset(new PosixRandomAccessFile(fname, fd));
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const std::string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  Status s = Status::OK;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = ErrnoToStatus(errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  std::string translated_name = TranslateName(fname);
  Status s = Status::OK;
  FILE* f = fopen(translated_name.c_str(), "a");
  if (f == nullptr) {
    s = ErrnoToStatus(errno);
  } else {
    result->reset(new PosixWritableFile(translated_name, f));
  }
  return s;
}

Status PosixFileSystem::FileExists(const std::string& fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) { return Status::OK; }
  return Status::NOT_FOUND;
}

Status PosixFileSystem::GetChildren(const std::string& dir,
                                    std::vector<std::string>* result) {
  std::string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) { return ErrnoToStatus(errno); }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    result->push_back(entry->d_name);
  }
  closedir(d);
  return Status::OK;
}

Status PosixFileSystem::DeleteFile(const std::string& fname) {
  Status s = Status::OK;
  if (unlink(TranslateName(fname).c_str()) != 0) { s = ErrnoToStatus(errno); }
  return s;
}

Status PosixFileSystem::CreateDir(const std::string& dirname) {
  Status s = Status::OK;
  if (mkdir(TranslateName(dirname).c_str(), 0755) != 0) {
    s = ErrnoToStatus(errno);
  }
  return s;
}

Status PosixFileSystem::DeleteDir(const std::string& dirname) {
  Status s = Status::OK;
  if (rmdir(TranslateName(dirname).c_str()) != 0) { s = ErrnoToStatus(errno); }
  return s;
}

Status PosixFileSystem::GetFileSize(const std::string& fname,
                                    uint64_t* file_size) {
  Status s = Status::OK;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *file_size = 0;
    s = ErrnoToStatus(errno);
  } else {
    *file_size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::RenameFile(const std::string& src,
                                   const std::string& target) {
  Status s = Status::OK;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    s = ErrnoToStatus(errno);
  }
  return s;
}

Status PosixFileSystem::IsDirectory(const std::string& fname) {
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) == 0 && S_ISDIR(sbuf.st_mode)) {
    return Status::OK;
  } else {
    return ErrnoToStatus(errno);
  }
}

}  // namespace fs

}  // namespace oneflow
