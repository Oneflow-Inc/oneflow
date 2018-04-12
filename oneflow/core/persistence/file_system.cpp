#include "oneflow/core/persistence/file_system.h"
#include <errno.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/persistence/windows/windows_file_system.h"

namespace oneflow {

namespace fs {

void FileSystem::CreateDirIfNotExist(const std::string& dirname) {
  if (IsDirectory(dirname)) { return; }
  CreateDir(dirname);
}

void FileSystem::RecursivelyCreateDirIfNotExist(const std::string& dirname) {
  if (IsDirectory(dirname)) { return; }
  RecursivelyCreateDir(dirname);
}

bool FileSystem::IsDirEmpty(const std::string& dirname) {
  return ListDir(dirname).empty();
}

std::string FileSystem::TranslateName(const std::string& name) const {
  return CleanPath(name);
}

void FileSystem::MakeEmptyDir(const std::string& dirname) {
  if (IsDirectory(dirname)) { RecursivelyDeleteDir(dirname); }
  RecursivelyCreateDir(dirname);
}

void FileSystem::RecursivelyDeleteDir(const std::string& dirname) {
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
    // GetChildren might fail if we don't have appropriate permissions.
    std::vector<std::string> children = ListDir(dir);
    for (const std::string& child : children) {
      const std::string child_path = JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path)) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        DelFile(child_path);
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
    CreateDir(built_path);
  }
}

struct GlobalFSConstructor {
  GlobalFSConstructor() {
    const GlobalFSConf& gfs_conf =
        Global<JobDesc>::Get()->job_conf().globalfs_conf();
    if (gfs_conf.has_localfs_conf()) {
      CHECK_EQ(Global<JobDesc>::Get()->resource().machine().size(), 1);
      gfs = LocalFS();
    } else if (gfs_conf.has_hdfs_conf()) {
      gfs = new HadoopFileSystem(gfs_conf.hdfs_conf());
    } else {
      UNIMPLEMENTED();
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
