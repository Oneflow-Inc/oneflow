#ifndef ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_

#include "oneflow/core/persistence/file_system.h"

extern "C" {
  struct hdfs_internal;
  typedef hdfs_internal* hdfsFS;
}

namespace oneflow {

namespace fs {

class LibHDFS;

class HadoopFileSystem final : public FileSystem {
public:
  OF_DISALLOW_COPY_AND_MOVE(HadoopFileSystem);
  HadoopFileSystem() = default;
  ~HadoopFileSystem() = default;

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
  Status Connect(std::string fname, hdfsFS* fs);
  LibHDFS* hdfs_;
};

}  // namespace fs

}  // namespace oneflow


#endif  // ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_
