#include "oneflow/core/persistence/file_system.h"
#include <errno.h>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/persistence/windows/windows_file_system.h"

namespace oneflow {

namespace fs {

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(Status);

void FileSystem::CreateDirIfNotExist(const std::string& dirname) {
  if (IsDirectory(dirname) == Status::OK) { return; }
  FS_CHECK_OK(CreateDir(dirname));
}

bool FileSystem::IsDirEmpty(const std::string& dirname) {
  return GetChildrenNumOfDir(dirname) == 0;
}

size_t FileSystem::GetChildrenNumOfDir(const std::string& dirname) {
  std::vector<std::string> result;
  FS_CHECK_OK(GetChildren(dirname, &result));
  return result.size();
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

Status FileSystem::DeleteRecursively(const std::string& dirname) {
  FS_CHECK_OK(FileExists(dirname));
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
    TryUpdateStatus(&ret, s);
    FS_CHECK_OK(s);
    for (const std::string& child : children) {
      const std::string child_path = JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path) == Status::OK) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        TryUpdateStatus(&ret, del_status);
        CHECK_EQ(del_status, Status::OK);
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
    TryUpdateStatus(&ret, s);
    FS_CHECK_OK(s);
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

void TryUpdateStatus(Status* current_status, const Status& new_status) {
  if (*current_status == Status::OK) { *current_status = new_status; }
}

Status ErrnoToStatus(int err_number) {
  Status ret;
  switch (err_number) {
    case 0: ret = Status::OK; break;
    case EINVAL:        // Invalid argument
    case ENAMETOOLONG:  // Filename too long
    case E2BIG:         // Argument list too long
    case EDESTADDRREQ:  // Destination address required
    case EDOM:          // Mathematics argument out of domain of function
    case EFAULT:        // Bad address
    case EILSEQ:        // Illegal byte sequence
    case ENOPROTOOPT:   // Protocol not available
    case ENOSTR:        // Not a STREAM
    case ENOTSOCK:      // Not a socket
    case ENOTTY:        // Inappropriate I/O control operation
    case EPROTOTYPE:    // Protocol wrong type for socket
    case ESPIPE:        // Invalid seek
      ret = Status::INVALID_ARGUMENT;
      break;
    case ETIMEDOUT:  // Connection timed out
    case ETIME:      // Timer expired
      ret = Status::DEADLINE_EXCEEDED;
      break;
    case ENODEV:  // No such device
    case ENOENT:  // No such file or directory
    case ENXIO:   // No such device or address
    case ESRCH:   // No such process
      ret = Status::NOT_FOUND;
      break;
    case EEXIST:         // File exists
    case EADDRNOTAVAIL:  // Address not available
    case EALREADY:       // Connection already in progress
      ret = Status::ALREADY_EXISTS;
      break;
    case EPERM:   // Operation not permitted
    case EACCES:  // Permission denied
    case EROFS:   // Read only file system
      ret = Status::PERMISSION_DENIED;
      break;
    case ENOTEMPTY:   // Directory not empty
    case EISDIR:      // Is a directory
    case ENOTDIR:     // Not a directory
    case EADDRINUSE:  // Address already in use
    case EBADF:       // Invalid file descriptor
    case EBUSY:       // Device or resource busy
    case ECHILD:      // No child processes
    case EISCONN:     // Socket is connected
#if !defined(_WIN32)
    case ENOTBLK:  // Block device required
#endif
    case ENOTCONN:  // The socket is not connected
    case EPIPE:     // Broken pipe
#if !defined(_WIN32)
    case ESHUTDOWN:  // Cannot send after transport endpoint shutdown
#endif
    case ETXTBSY:  // Text file busy
      ret = Status::FAILED_PRECONDITION;
      break;
    case ENOSPC:  // No space left on device
#if !defined(_WIN32)
    case EDQUOT:  // Disk quota exceeded
#endif
    case EMFILE:   // Too many open files
    case EMLINK:   // Too many links
    case ENFILE:   // Too many open files in system
    case ENOBUFS:  // No buffer space available
    case ENODATA:  // No message is available on the STREAM read queue
    case ENOMEM:   // Not enough space
    case ENOSR:    // No STREAM resources
#if !defined(_WIN32)
    case EUSERS:  // Too many users
#endif
      ret = Status::RESOURCE_EXHAUSTED;
      break;
    case EFBIG:      // File too large
    case EOVERFLOW:  // Value too large to be stored in data type
    case ERANGE:     // Result too large
      ret = Status::OUT_OF_RANGE;
      break;
    case ENOSYS:        // Function not implemented
    case ENOTSUP:       // Operation not supported
    case EAFNOSUPPORT:  // Address family not supported
#if !defined(_WIN32)
    case EPFNOSUPPORT:  // Protocol family not supported
#endif
    case EPROTONOSUPPORT:  // Protocol not supported
#if !defined(_WIN32)
    case ESOCKTNOSUPPORT:  // Socket type not supported
#endif
    case EXDEV:  // Improper link
      ret = Status::UNIMPLEMENTED;
      break;
    case EAGAIN:        // Resource temporarily unavailable
    case ECONNREFUSED:  // Connection refused
    case ECONNABORTED:  // Connection aborted
    case ECONNRESET:    // Connection reset
    case EINTR:         // Interrupted function call
#if !defined(_WIN32)
    case EHOSTDOWN:  // Host is down
#endif
    case EHOSTUNREACH:  // Host is unreachable
    case ENETDOWN:      // Network is down
    case ENETRESET:     // Connection aborted by network
    case ENETUNREACH:   // Network unreachable
    case ENOLCK:        // No locks available
    case ENOLINK:       // Link has been severed
#if !defined(_WIN32)
    case ENONET:  // Machine is not on the network
#endif
      ret = Status::UNAVAILABLE;
      break;
    case EDEADLK:  // Resource deadlock avoided
#if !defined(_WIN32)
    case ESTALE:  // Stale file handle
#endif
      ret = Status::ABORTED;
      break;
    case ECANCELED:  // Operation cancelled
      ret = Status::CANCELLED;
      break;
    // NOTE: If you get any of the following (especially in a
    // reproducible way) and can propose a better mapping,
    // please email the owners about updating this mapping.
    case EBADMSG:      // Bad message
    case EIDRM:        // Identifier removed
    case EINPROGRESS:  // Operation in progress
    case EIO:          // I/O Status
    case ELOOP:        // Too many levels of symbolic links
    case ENOEXEC:      // Exec format Status
    case ENOMSG:       // No message of the desired type
    case EPROTO:       // Protocol Status
#if !defined(_WIN32)
    case EREMOTE:  // Object is remote
#endif
      ret = Status::UNKNOWN;
      break;
    default: {
      ret = Status::UNKNOWN;
      break;
    }
  }
  return ret;
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
