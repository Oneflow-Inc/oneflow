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

  void Read(uint64_t offset, size_t n, char* result) const override {
    char* dst = result;
    while (n > 0) {
      ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        PLOG(FATAL) << "Read EOF";
        return;
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        PLOG(FATAL) << "Fail to read file " << fname_;
        return;
      }
    }
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

  void Append(const char* data, size_t n) override {
    PCHECK(fwrite(data, sizeof(char), n, file_) == n)
        << "Fail to append to file " << fname_;
  }

  void Close() override {
    Flush();
    PCHECK(fclose(file_) == 0) << "Fail to close file " << fname_;
    file_ = nullptr;
  }

  void Flush() override {
    PCHECK(fflush(file_) == 0) << "Fail to flush file " << fname_;
  }
};

void PosixFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  std::string translated_fname = TranslateName(fname);
  int fd = open(translated_fname.c_str(), O_RDONLY);
  PCHECK(fd >= 0) << "Fail to open file " << fname;
  result->reset(new PosixRandomAccessFile(fname, fd));
  CHECK_NOTNULL(result->get());
}

void PosixFileSystem::NewWritableFile(const std::string& fname,
                                      std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  FILE* f = fopen(translated_fname.c_str(), "w");
  PCHECK(f != nullptr) << "Fail to open file " << fname;
  result->reset(new PosixWritableFile(translated_fname, f));
  CHECK_NOTNULL(result->get());
}

void PosixFileSystem::NewAppendableFile(const std::string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  std::string translated_name = TranslateName(fname);
  FILE* f = fopen(translated_name.c_str(), "a");
  PCHECK(f != nullptr) << "Fail to open file " << fname;
  result->reset(new PosixWritableFile(translated_name, f));
  CHECK_NOTNULL(result->get());
}

bool PosixFileSystem::FileExists(const std::string& fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) { return true; }
  return false;
}

std::vector<std::string> PosixFileSystem::ListDir(const std::string& dir) {
  std::string translated_dir = TranslateName(dir);
  std::vector<std::string> result;
  DIR* d = opendir(translated_dir.c_str());
  PCHECK(d != nullptr) << "Fail to open dir " << dir;
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    result.push_back(entry->d_name);
  }
  closedir(d);
  return result;
}

void PosixFileSystem::DelFile(const std::string& fname) {
  PCHECK(unlink(TranslateName(fname).c_str()) == 0)
      << "Fail to delete file " << fname;
}

void PosixFileSystem::CreateDir(const std::string& dirname) {
  PCHECK(mkdir(TranslateName(dirname).c_str(), 0755) == 0)
      << "Fail to create dir " << dirname;
}

void PosixFileSystem::DeleteDir(const std::string& dirname) {
  PCHECK(rmdir(TranslateName(dirname).c_str()) == 0)
      << "Fail to delete dir " << dirname;
}

uint64_t PosixFileSystem::GetFileSize(const std::string& fname) {
  struct stat sbuf;
  PCHECK(stat(TranslateName(fname).c_str(), &sbuf) == 0)
      << "Fail to load statistics of " << fname;
  ;
  return sbuf.st_size;
}

void PosixFileSystem::RenameFile(const std::string& old_name,
                                 const std::string& new_name) {
  PCHECK(
      rename(TranslateName(old_name).c_str(), TranslateName(new_name).c_str())
      == 0)
      << "Fail to rename file from " << old_name << " to " << new_name;
}

bool PosixFileSystem::IsDirectory(const std::string& fname) {
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) == 0 && S_ISDIR(sbuf.st_mode)) {
    return true;
  }
  return false;
}

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_POSIX
