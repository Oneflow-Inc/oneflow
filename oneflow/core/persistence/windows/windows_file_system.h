#ifndef ONEFLOW_CORE_PERSISTENCE_WINDOWS_WINDOWS_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_WINDOWS_WINDOWS_FILE_SYSTEM_H_

#ifdef PLATFORM_WINDOWS

#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

namespace fs {

class WindowsFileSystem final : public FileSystem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WindowsFileSystem);
  WindowsFileSystem() = default;
  ~WindowsFileSystem() = default;

  Status NewRandomAccessFile(
      const std::string& fname,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const std::string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status FileExists(const std::string& fname) override;

  Status GetChildren(const std::string& dir,
                     std::vector<std::string>* result) override;

  Status DeleteFile(const std::string& fname) override;

  Status CreateDir(const std::string& dirname) override;

  Status DeleteDir(const std::string& dirname) override;

  Status GetFileSize(const std::string& fname, uint64_t* file_size) override;

  Status RenameFile(const std::string& src, const std::string& target) override;

  Status IsDirectory(const std::string& fname) override;

 private:
};

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS

#endif  // ONEFLOW_CORE_PERSISTENCE_WINDOWS_WINDOWS_FILE_SYSTEM_H_
