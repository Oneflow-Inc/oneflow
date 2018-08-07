#ifndef ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "hdfs/hdfs.h"

extern "C" {
struct HdfsFileSystemInternalWrapper;
typedef HdfsFileSystemInternalWrapper* hdfsFS;
}

namespace oneflow {

namespace fs {

class LibHDFS {
 public:
  static LibHDFS* Load() {
    LibHDFS* lib = new LibHDFS();
    return lib;
  }

  std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect = hdfsBuilderConnect;
  std::function<hdfsBuilder*()> hdfsNewBuilder = hdfsNewBuilder;
  std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode = hdfsBuilderSetNameNode;
  std::function<int(const char*, char**)> hdfsConfGetStr = hdfsConfGetStr;
  std::function<void(hdfsBuilder*, const char* kerbTicketCachePath)>
      hdfsBuilderSetKerbTicketCachePath = hdfsBuilderSetKerbTicketCachePath;
  std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile = hdfsCloseFile;
  std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread = hdfsPread;
  std::function<tSize(hdfsFS, hdfsFile, const void*, tSize)> hdfsWrite = hdfsWrite;
  std::function<int(hdfsFS, hdfsFile)> hdfsHFlush = hdfsHFlush;
  std::function<int(hdfsFS, hdfsFile)> hdfsHSync = hdfsHSync;
  std::function<hdfsFile(hdfsFS, const char*, int, int, short, tSize)> hdfsOpenFile = hdfsOpenFile;
  std::function<int(hdfsFS, const char*)> hdfsExists = hdfsExists;
  std::function<hdfsFileInfo*(hdfsFS, const char*, int*)> hdfsListDirectory = hdfsListDirectory;
  std::function<void(hdfsFileInfo*, int)> hdfsFreeFileInfo = hdfsFreeFileInfo;
  std::function<int(hdfsFS, const char*, int recursive)> hdfsDelete = hdfsDelete;
  std::function<int(hdfsFS, const char*)> hdfsCreateDirectory = hdfsCreateDirectory;
  std::function<hdfsFileInfo*(hdfsFS, const char*)> hdfsGetPathInfo = hdfsGetPathInfo;
  std::function<int(hdfsFS, const char*, const char*)> hdfsRename = hdfsRename;
};

class HadoopFileSystem final : public FileSystem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HadoopFileSystem);
  HadoopFileSystem() = delete;
  ~HadoopFileSystem() = default;

  HadoopFileSystem(const HdfsConf&);

  void NewRandomAccessFile(const std::string& fname,
                           std::unique_ptr<RandomAccessFile>* result) override;

  void NewWritableFile(const std::string& fname, std::unique_ptr<WritableFile>* result) override;

  void NewAppendableFile(const std::string& fname, std::unique_ptr<WritableFile>* result) override;

  bool FileExists(const std::string& fname) override;

  std::vector<std::string> ListDir(const std::string& dir) override;

  void DelFile(const std::string& fname) override;

  void CreateDir(const std::string& dirname) override;

  void DeleteDir(const std::string& dirname) override;

  void RecursivelyDeleteDir(const std::string& dirname) override;

  uint64_t GetFileSize(const std::string& fname) override;

  void RenameFile(const std::string& old_name, const std::string& new_name) override;

  bool IsDirectory(const std::string& fname) override;

 private:
  bool Connect(hdfsFS* fs);
  std::string namenode_;
  LibHDFS* hdfs_;
};

}  // namespace fs

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_HADOOP_HADOOP_FILE_SYSTEM_H_
