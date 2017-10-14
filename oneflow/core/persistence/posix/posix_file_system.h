#ifndef ONEFLOW_CORE_PERSISTENCE_POSIX_POSIX_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_POSIX_POSIX_FILE_SYSTEM_H_

#include "oneflow/core/persistence/file_system.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

namespace fs {

class PosixFileSystem final : public FileSystem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PosixFileSystem);
  PosixFileSystem() = default;
  ~PosixFileSystem() = default;

  void NewRandomAccessFile(const std::string& fname,
                           std::unique_ptr<RandomAccessFile>* result) override;

  void NewWritableFile(const std::string& fname,
                       std::unique_ptr<WritableFile>* result) override;

  void NewAppendableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  bool FileExists(const std::string& fname) override;

  std::vector<std::string> ListDir(const std::string& dir) override;

  void DelFile(const std::string& fname) override;

  void CreateDir(const std::string& dirname) override;

  void DeleteDir(const std::string& dirname) override;

  uint64_t GetFileSize(const std::string& fname) override;

  void RenameFile(const std::string& old_name,
                  const std::string& new_name) override;

  bool IsDirectory(const std::string& fname) override;

 private:
};

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_PERSISTENCE_POSIX_POSIX_FILE_SYSTEM_H_
