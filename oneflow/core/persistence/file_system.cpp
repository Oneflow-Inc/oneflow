#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

std::string StatusToString(Status s) {
  std::string result;
  switch (s) {
    case Status::OK: result = "OK"; break;
    case Status::FAILED_PRECONDITION: result = "Failed precondition"; break;
    case Status::NOT_FOUND: result = "Already exists"; break;
    case Status::ALREADY_EXISTS: result = "Not found"; break;
    case Status::PERMISSION_DENIED: result = "Permission denied"; break;
    case Status::UNIMPLEMENTED: result = "Unimplemented"; break;
    default: result = "Unknown code " + std::to_string(s); break;
  }
  return result;
}

std::string FileSystem::TranslateName(const std::string& name) const {
  return CleanPath(name);
}

bool FileSystem::FilesExist(const std::vector<std::string>& files,
                            std::vector<Status>* status) {
  bool result = true;
  for (const auto& file : files) {
    Status s = FileExists(file);
    result &= (s == Status::OK);
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      // Return early since there is no need to check other files.
      return false;
    }
  }
  return result;
}

Status FileSystem::DeleteRecursively(const std::string& dirname,
                                     int64_t* undeleted_files,
                                     int64_t* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  Status exists_status = FileExists(dirname);
  if (exists_status != Status::OK) {
    (*undeleted_dirs)++;
    return exists_status;
  }
  std::deque<std::string> dir_q;      // Queue for the BFS
  std::vector<std::string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  // ret : Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  Status ret = Status::OK;
  while (!dir_q.empty()) {
    std::string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<std::string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    Status s = GetChildren(dir, &children);
    // update ret
    if (ret == Status::OK) { ret = s; }
    if (s != Status::OK) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const std::string& child : children) {
      const std::string child_path = JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path) == Status::OK) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        // update ret
        if (ret == Status::OK) { ret = del_status; }
        if (del_status != Status::OK) { (*undeleted_files)++; }
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const std::string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    Status s = DeleteDir(dir);
    // update ret
    if (ret == Status::OK) { ret = s; }
    if (s != Status::OK) { (*undeleted_dirs)++; }
  }
  return ret;
}

Status FileSystem::RecursivelyCreateDir(const std::string& dirname) {
  std::string remaining_dir = dirname;
  std::vector<std::string> sub_dirs;
  while (!remaining_dir.empty()) {
    Status status = FileExists(remaining_dir);
    if (status == Status::OK) { break; }
    if (status != Status::NOT_FOUND) { return status; }
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
    Status status = CreateDir(built_path);
    if (status != Status::OK && status != Status::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK;
}

}  // namespace oneflow
