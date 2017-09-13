#ifndef ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/hadoop/hdfs.h"

extern "C" {
struct hdfs_internal;
typedef hdfs_internal* hdfsFS;
}

namespace oneflow {

namespace fs {

class LibHDFS {
 public:
  static LibHDFS* Load() {
    static LibHDFS* lib = []() -> LibHDFS* {
      LibHDFS* lib = new LibHDFS;
      lib->LoadAndBind();
      return lib;
    }();
    return lib;
  }

  // The status, if any, from failure to load.
  bool status() { return status_; }

  std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect;
  std::function<hdfsBuilder*()> hdfsNewBuilder;
  std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode;
  std::function<int(const char*, char**)> hdfsConfGetStr;
  std::function<void(hdfsBuilder*, const char* kerbTicketCachePath)>
      hdfsBuilderSetKerbTicketCachePath;
  std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile;
  std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread;
  std::function<tSize(hdfsFS, hdfsFile, const void*, tSize)> hdfsWrite;
  std::function<int(hdfsFS, hdfsFile)> hdfsHFlush;
  std::function<int(hdfsFS, hdfsFile)> hdfsHSync;
  std::function<hdfsFile(hdfsFS, const char*, int, int, short, tSize)>
      hdfsOpenFile;
  std::function<int(hdfsFS, const char*)> hdfsExists;
  std::function<hdfsFileInfo*(hdfsFS, const char*, int*)> hdfsListDirectory;
  std::function<void(hdfsFileInfo*, int)> hdfsFreeFileInfo;
  std::function<int(hdfsFS, const char*, int recursive)> hdfsDelete;
  std::function<int(hdfsFS, const char*)> hdfsCreateDirectory;
  std::function<hdfsFileInfo*(hdfsFS, const char*)> hdfsGetPathInfo;
  std::function<int(hdfsFS, const char*, const char*)> hdfsRename;

 private:
  void LoadAndBind();
  bool status_;
  void* handle_ = nullptr;
};

class HadoopFileSystem final : public FileSystem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HadoopFileSystem);
  HadoopFileSystem() = delete;
  ~HadoopFileSystem() = default;

  HadoopFileSystem(const HdfsConf&);

  void NewRandomAccessFile(const std::string& fname,
                           std::unique_ptr<RandomAccessFile>* result) override;

  void NewWritableFile(const std::string& fname,
                       std::unique_ptr<WritableFile>* result) override;

  void NewAppendableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  bool FileExists(const std::string& fname) override;

  std::vector<std::string> ListDir(const std::string& dir) override;

  void DeleteFile(const std::string& fname) override;

  void CreateDir(const std::string& dirname) override;

  void DeleteDir(const std::string& dirname) override;

  uint64_t GetFileSize(const std::string& fname) override;

  void RenameFile(const std::string& old_name,
                  const std::string& new_name) override;

  bool IsDirectory(const std::string& fname) override;

 private:
  bool Connect(hdfsFS* fs);
  std::string namenode_;
  LibHDFS* hdfs_;
};

}  // namespace fs

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_
