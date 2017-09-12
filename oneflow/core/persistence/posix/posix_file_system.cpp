#include "oneflow/core/persistence/posix/posix_file_system.h"

#ifdef PLATFORM_POSIX

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

  size_t Read(uint64_t offset, size_t n, char* result) const override {
    char* dst = result;
    size_t read_count = 0;
    while (n > 0) {
      ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
        read_count += r;
      } else if (r == 0) {
        LOG(FATAL) << "OUT OF RANGE";
        return read_count;
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        LOG(FATAL) << "FAIL TO READ FILE";
        return read_count;
      }
    }
    return read_count;
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

  size_t Append(const char* data, size_t n) override {
    return fwrite(data, sizeof(char), n, file_);
  }

  void Close() override {
    Flush();
    if (fclose(file_) != 0) { LOG(FATAL) << "FAIL TO CLOSE FILE"; }
    file_ = nullptr;
  }

  void Flush() override {
    if (fflush(file_) != 0) { LOG(FATAL) << "FAIL TO FLUSH FILE"; }
  }
};

void PosixFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  std::string translated_fname = TranslateName(fname);
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    LOG(FATAL) << "FAIL TO OPEN FILE";
  } else {
    result->reset(new PosixRandomAccessFile(fname, fd));
  }
}

void PosixFileSystem::NewWritableFile(const std::string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    LOG(FATAL) << "FAIL TO OPEN FILE";
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
}

void PosixFileSystem::NewAppendableFile(const std::string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  std::string translated_name = TranslateName(fname);
  FILE* f = fopen(translated_name.c_str(), "a");
  if (f == nullptr) {
    LOG(FATAL) << "FAIL TO OPEN FILE";
  } else {
    result->reset(new PosixWritableFile(translated_name, f));
  }
}

bool PosixFileSystem::FileExists(const std::string& fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) { return true; }
  return false;
}

void PosixFileSystem::GetChildren(const std::string& dir,
                                  std::vector<std::string>* result) {
  std::string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) { LOG(FATAL) << "FAIL TO OPEN DIR"; }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    result->push_back(entry->d_name);
  }
  closedir(d);
}

void PosixFileSystem::DeleteFile(const std::string& fname) {
  if (unlink(TranslateName(fname).c_str()) != 0) {
    LOG(FATAL) << "FAIL TO DELETE FILE";
  }
}

bool PosixFileSystem::CreateDir(const std::string& dirname) {
  if (mkdir(TranslateName(dirname).c_str(), 0755) != 0) { return false; }
  return true;
}

void PosixFileSystem::DeleteDir(const std::string& dirname) {
  if (rmdir(TranslateName(dirname).c_str()) != 0) {
    LOG(FATAL) << "FAIL TO DELETE DIR";
  }
}

void PosixFileSystem::GetFileSize(const std::string& fname,
                                  uint64_t* file_size) {
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *file_size = 0;
    LOG(FATAL) << "FAIL TO LOAD FILE STATISTICS";
  } else {
    *file_size = sbuf.st_size;
  }
}

void PosixFileSystem::RenameFile(const std::string& src,
                                 const std::string& target) {
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    LOG(FATAL) << "FAIL TO RENAME FILE";
  }
}

bool PosixFileSystem::IsDirectory(const std::string& fname) {
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) == 0 && S_ISDIR(sbuf.st_mode)) {
    return true;
  } else {
    return false;
  }
}

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_POSIX
