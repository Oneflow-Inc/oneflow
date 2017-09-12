#include "oneflow/core/persistence/file_system.h"
#include <errno.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/persistence/windows/windows_file_system.h"

namespace oneflow {

namespace fs {

void FileSystem::CreateDirIfNotExist(const std::string& dirname) {
  if (IsDirectory(dirname)) { return; }
  CHECK(CreateDir(dirname));
}

bool FileSystem::IsDirEmpty(const std::string& dirname) {
  return GetChildrenNumOfDir(dirname) == 0;
}

size_t FileSystem::GetChildrenNumOfDir(const std::string& dirname) {
  std::vector<std::string> result;
  GetChildren(dirname, &result);
  return result.size();
}

std::string FileSystem::TranslateName(const std::string& name) const {
  return CleanPath(name);
}

bool FileSystem::FilesExist(const std::vector<std::string>& files,
                            std::vector<bool>* ret) {
  bool result = true;
  for (const auto& file : files) {
    bool s = FileExists(file);
    result &= s;
    if (ret != nullptr) { ret->push_back(s); }
  }
  return result;
}

void FileSystem::DeleteRecursively(const std::string& dirname) {
  CHECK(FileExists(dirname));
  std::deque<std::string> dir_q;      // Queue for the BFS
  std::vector<std::string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  // ret : Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    std::string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<std::string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    GetChildren(dir, &children);
    for (const std::string& child : children) {
      const std::string child_path = JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path)) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        DeleteFile(child_path);
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const std::string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    DeleteDir(dir);
  }
}

void FileSystem::RecursivelyCreateDir(const std::string& dirname) {
  std::string remaining_dir = dirname;
  std::vector<std::string> sub_dirs;
  while (!remaining_dir.empty()) {
    bool status = FileExists(remaining_dir);
    if (status) { break; }
    // Basename returns "" for / ending dirs.
    if (remaining_dir[remaining_dir.length() - 1] != '/') {
      sub_dirs.push_back(Basename(remaining_dir));
    }
    remaining_dir = Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  std::string built_path = remaining_dir;
  for (const std::string& sub_dir : sub_dirs) {
    built_path = JoinPath(built_path, sub_dir);
    CHECK(CreateDir(built_path));
  }
}

struct GlobalFSConstructor {
  GlobalFSConstructor() {
    const GlobalFSConf& gfs_conf =
        JobDesc::Singleton()->job_conf().global_fs_conf();
    if (gfs_conf.has_localfs_conf()) {
      CHECK_EQ(JobDesc::Singleton()->resource().machine().size(), 1);
      gfs = LocalFS();
    } else if (gfs_conf.has_hdfs_conf()) {
      // static fs::FileSystem* fs = new
      // fs::HadoopFileSystem(gfs_conf.hdfs_conf()); return fs;
    } else {
      UNEXPECTED_RUN();
    }
  }
  FileSystem* gfs;
};

}  // namespace fs

fs::FileSystem* LocalFS() {
#ifdef PLATFORM_POSIX
  static fs::FileSystem* fs = new fs::PosixFileSystem;
#elif PLATFORM_WINDOWS
  static fs::FileSystem* fs = new fs::WindowsFileSystem;
#endif
  return fs;
}

fs::FileSystem* GlobalFS() {
  static fs::GlobalFSConstructor gfs_constructor;
  return gfs_constructor.gfs;
}

}  // namespace oneflow
