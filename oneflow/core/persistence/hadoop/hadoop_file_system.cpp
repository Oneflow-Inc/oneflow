#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include <mutex>
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/windows/windows_file_system.h"

#ifdef PLATFORM_WINDOWS

#undef LoadLibrary

#endif  // PLATFORM_WINDOWS

#ifdef PLATFORM_POSIX

#include <dlfcn.h>

#endif  // PLATFORM_POSIX

namespace oneflow {

namespace fs {

namespace internal {

#ifdef PLATFORM_POSIX

Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol) {
  *symbol = dlsym(handle, symbol_name);
  if (!*symbol) {
    PLOG(WARNING) << dlerror();
    return Status::NOT_FOUND;
  }
  return Status::OK;
}

Status LoadLibrary(const char* library_filename, void** handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    PLOG(WARNING) << dlerror();
    return Status::NOT_FOUND;
  }
  return Status::OK;
}

#endif  // PLATFORM_POSIX

#ifdef PLATFORM_WINDOWS

Status LoadLibrary(const char* library_filename, void** handle) {
  std::string file_name = library_filename;
  std::replace(file_name.begin(), file_name.end(), '/', '\\');

  std::wstring ws_file_name(WindowsFileSystem::Utf8ToWideChar(file_name));

  HMODULE hModule =
      LoadLibraryExW(ws_file_name.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!hModule) {
    PLOG(WARNING) << file_name + "not found";
    return Status::NOT_FOUND;
  }
  *handle = hModule;
  return Status::OK;
}

Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                            void** symbol) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);
  if (found_symbol == NULL) {
    PLOG(WARNING) << std::string(symbol_name) + "not found";
    return Status::NOT_FOUND;
  }
  *symbol = (void**)found_symbol;
  return Status::OK;
}

#endif  // PLATFORM_WINDOWS

}  // namespace internal

template<typename R, typename... Args>
Status BindFunc(void* handle, const char* name,
                std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;
  FS_RETURN_IF_ERR(internal::GetSymbolFromLibrary(handle, name, &symbol_ptr));
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return Status::OK;
}

void LibHDFS::LoadAndBind() {
  auto TryLoadAndBind = [this](const char* name, void** handle) -> Status {
    FS_RETURN_IF_ERR(internal::LoadLibrary(name, handle));
#define BIND_HDFS_FUNC(function) \
  FS_RETURN_IF_ERR(BindFunc(*handle, #function, &function));

    BIND_HDFS_FUNC(hdfsBuilderConnect);
    BIND_HDFS_FUNC(hdfsNewBuilder);
    BIND_HDFS_FUNC(hdfsBuilderSetNameNode);
    BIND_HDFS_FUNC(hdfsConfGetStr);
    BIND_HDFS_FUNC(hdfsBuilderSetKerbTicketCachePath);
    BIND_HDFS_FUNC(hdfsCloseFile);
    BIND_HDFS_FUNC(hdfsPread);
    BIND_HDFS_FUNC(hdfsWrite);
    BIND_HDFS_FUNC(hdfsHFlush);
    BIND_HDFS_FUNC(hdfsHSync);
    BIND_HDFS_FUNC(hdfsOpenFile);
    BIND_HDFS_FUNC(hdfsExists);
    BIND_HDFS_FUNC(hdfsListDirectory);
    BIND_HDFS_FUNC(hdfsFreeFileInfo);
    BIND_HDFS_FUNC(hdfsDelete);
    BIND_HDFS_FUNC(hdfsCreateDirectory);
    BIND_HDFS_FUNC(hdfsGetPathInfo);
    BIND_HDFS_FUNC(hdfsRename);
#undef BIND_HDFS_FUNC
    return Status::OK;
  };

// libhdfs.so won't be in the standard locations. Use the path as specified
// in the libhdfs documentation.
#if defined(PLATFORM_WINDOWS)
  const char* kLibHdfsDso = "hdfs.dll";
#else
  const char* kLibHdfsDso = "libhdfs.so";
#endif
  char* hdfs_home = getenv("HADOOP_HDFS_HOME");
  if (hdfs_home == nullptr) {
    PLOG(WARNING) << "Environment variable HADOOP_HDFS_HOME not set";
    status_ = Status::FAILED_PRECONDITION;
    return;
  }
  std::string path = JoinPath(hdfs_home, "lib", "native", kLibHdfsDso);
  status_ = TryLoadAndBind(path.c_str(), &handle_);
  if (status_ != Status::OK) {
    // try load libhdfs.so using dynamic loader's search path in case
    // libhdfs.so is installed in non-standard location
    status_ = TryLoadAndBind(kLibHdfsDso, &handle_);
  }
}

HadoopFileSystem::HadoopFileSystem(const HdfsConf& hdfs_conf)
    : namenode_(hdfs_conf.namenode()), hdfs_(LibHDFS::Load()) {}

Status HadoopFileSystem::Connect(hdfsFS* fs) {
  FS_RETURN_IF_ERR(hdfs_->status());
  hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
  hdfs_->hdfsBuilderSetNameNode(builder, namenode_.c_str());
  // KERB_TICKET_CACHE_PATH will be deleted in the future, Because KRB5CCNAME is
  // the build in environment variable of Kerberos, so KERB_TICKET_CACHE_PATH
  // and related code are unnecessary.
  char* ticket_cache_path = getenv("KERB_TICKET_CACHE_PATH");
  if (ticket_cache_path != nullptr) {
    hdfs_->hdfsBuilderSetKerbTicketCachePath(builder, ticket_cache_path);
  }
  *fs = hdfs_->hdfsBuilderConnect(builder);
  if (*fs == nullptr) {
    PLOG(WARNING) << " not found";
    return Status::NOT_FOUND;
  }
  return Status::OK;
}

class HDFSRandomAccessFile : public RandomAccessFile {
 public:
  HDFSRandomAccessFile(const std::string& filename,
                       const std::string& hdfs_filename, LibHDFS* hdfs,
                       hdfsFS fs, hdfsFile file)
      : filename_(filename),
        hdfs_filename_(hdfs_filename),
        hdfs_(hdfs),
        fs_(fs),
        file_(file) {}

  ~HDFSRandomAccessFile() override {
    if (file_ != nullptr) {
      std::unique_lock<std::mutex> lock(mu_);
      hdfs_->hdfsCloseFile(fs_, file_);
    }
  }

  Status Read(uint64_t offset, size_t n, char* result) const override {
    Status s = Status::OK;
    char* dst = result;
    bool eof_retried = false;
    while (n > 0 && s == Status::OK) {
      // We lock inside the loop rather than outside so we don't block other
      // concurrent readers.
      std::unique_lock<std::mutex> lock(mu_);
      tSize r = hdfs_->hdfsPread(fs_, file_, static_cast<tOffset>(offset), dst,
                                 static_cast<tSize>(n));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (!eof_retried && r == 0) {
        // Always reopen the file upon reaching EOF to see if there's more data.
        // If writers are streaming contents while others are concurrently
        // reading, HDFS requires that we reopen the file to see updated
        // contents.
        if (file_ != nullptr && hdfs_->hdfsCloseFile(fs_, file_) != 0) {
          PLOG(WARNING) << filename_;
          return ErrnoToStatus(errno);
        }
        file_ =
            hdfs_->hdfsOpenFile(fs_, hdfs_filename_.c_str(), O_RDONLY, 0, 0, 0);
        if (file_ == nullptr) {
          PLOG(WARNING) << filename_;
          return ErrnoToStatus(errno);
        }
        eof_retried = true;
      } else if (eof_retried && r == 0) {
        PLOG(WARNING) << "Read less bytes than requested";
        s = Status::OUT_OF_RANGE;
      } else if (errno == EINTR || errno == EAGAIN) {
        // hdfsPread may return EINTR too. Just retry.
      } else {
        PLOG(WARNING) << filename_;
        s = ErrnoToStatus(errno);
      }
    }
    return s;
  }

 private:
  std::string filename_;
  std::string hdfs_filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;

  mutable std::mutex mu_;
  mutable hdfsFile file_;
};

Status HadoopFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_RDONLY, 0, 0, 0);
  if (file == nullptr) {
    PLOG(WARNING) << fname;
    return ErrnoToStatus(errno);
  }
  result->reset(
      new HDFSRandomAccessFile(fname, TranslateName(fname), hdfs_, fs, file));
  return Status::OK;
}

class HDFSWritableFile : public WritableFile {
 public:
  HDFSWritableFile(const std::string& fname, LibHDFS* hdfs, hdfsFS fs,
                   hdfsFile file)
      : filename_(fname), hdfs_(hdfs), fs_(fs), file_(file) {}

  ~HDFSWritableFile() override {
    if (file_ != nullptr) { Close(); }
  }

  Status Append(const char* data, size_t n) override {
    if (hdfs_->hdfsWrite(fs_, file_, data, static_cast<tSize>(n)) == -1) {
      PLOG(WARNING) << filename_;
      return ErrnoToStatus(errno);
    }
    return Status::OK;
  }

  Status Close() override {
    Status result;
    if (hdfs_->hdfsCloseFile(fs_, file_) != 0) {
      PLOG(WARNING) << filename_;
      result = ErrnoToStatus(errno);
    }
    hdfs_ = nullptr;
    fs_ = nullptr;
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (hdfs_->hdfsHFlush(fs_, file_) != 0) {
      PLOG(WARNING) << filename_;
      return ErrnoToStatus(errno);
    }
    return Status::OK;
  }

 private:
  std::string filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;
  hdfsFile file_;
};

Status HadoopFileSystem::NewWritableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_WRONLY, 0, 0, 0);
  if (file == nullptr) {
    PLOG(WARNING) << fname;
    return ErrnoToStatus(errno);
  }
  result->reset(new HDFSWritableFile(fname, hdfs_, fs, file));
  return Status::OK;
}

Status HadoopFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  hdfsFile file = hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                                      O_WRONLY | O_APPEND, 0, 0, 0);
  if (file == nullptr) {
    PLOG(WARNING) << fname;
    return ErrnoToStatus(errno);
  }
  result->reset(new HDFSWritableFile(fname, hdfs_, fs, file));
  return Status::OK;
}

Status HadoopFileSystem::FileExists(const std::string& fname) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));
  if (hdfs_->hdfsExists(fs, TranslateName(fname).c_str()) == 0) {
    return Status::OK;
  }
  PLOG(WARNING) << fname;
  return Status::NOT_FOUND;
}

Status HadoopFileSystem::GetChildren(const std::string& dir,
                                     std::vector<std::string>* result) {
  result->clear();
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
  // check to verify the directory exists first.
  FS_RETURN_IF_ERR(IsDirectory(dir));

  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info == nullptr) {
    // Assume it's an empty directory.
    return Status::OK;
  }
  for (int i = 0; i < entries; i++) {
    result->push_back(Basename(info[i].mName));
  }
  hdfs_->hdfsFreeFileInfo(info, entries);
  return Status::OK;
}

Status HadoopFileSystem::DeleteFile(const std::string& fname) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  if (hdfs_->hdfsDelete(fs, TranslateName(fname).c_str(), /*recursive=*/0)
      != 0) {
    PLOG(WARNING) << fname;
    return ErrnoToStatus(errno);
  }
  return Status::OK;
}

Status HadoopFileSystem::CreateDir(const std::string& dir) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  if (hdfs_->hdfsCreateDirectory(fs, TranslateName(dir).c_str()) != 0) {
    PLOG(WARNING) << dir;
    return ErrnoToStatus(errno);
  }
  return Status::OK;
}

Status HadoopFileSystem::DeleteDir(const std::string& dir) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  // Count the number of entries in the directory, and only delete if it's
  // non-empty. This is consistent with the interface, but note that there's
  // a race condition where a file may be added after this check, in which
  // case the directory will still be deleted.
  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info != nullptr) { hdfs_->hdfsFreeFileInfo(info, entries); }
  // Due to HDFS bug HDFS-8407, we can't distinguish between an error and empty
  // folder, expscially for Kerberos enable setup, EAGAIN is quite common when
  // the call is actually successful. Check again by Stat.
  if (info == nullptr && errno != 0) { FS_RETURN_IF_ERR(IsDirectory(dir)); }

  if (entries > 0) {
    PLOG(WARNING) << dir << "Cannot delete a non-empty directory.";
    return Status::FAILED_PRECONDITION;
  }
  if (hdfs_->hdfsDelete(fs, TranslateName(dir).c_str(), /*recursive=*/1) != 0) {
    PLOG(WARNING) << dir;
    return ErrnoToStatus(errno);
  }
  return Status::OK;
}

Status HadoopFileSystem::GetFileSize(const std::string& fname, uint64_t* size) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr) {
    PLOG(WARNING) << fname;
    return ErrnoToStatus(errno);
  }
  *size = static_cast<uint64_t>(info->mSize);
  hdfs_->hdfsFreeFileInfo(info, 1);
  return Status::OK;
}

Status HadoopFileSystem::RenameFile(const std::string& src,
                                    const std::string& target) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  if (hdfs_->hdfsExists(fs, TranslateName(target).c_str()) == 0
      && hdfs_->hdfsDelete(fs, TranslateName(target).c_str(), /*recursive=*/0)
             != 0) {
    PLOG(WARNING) << target;
    return ErrnoToStatus(errno);
  }

  if (hdfs_->hdfsRename(fs, TranslateName(src).c_str(),
                        TranslateName(target).c_str())
      != 0) {
    PLOG(WARNING) << src;
    return ErrnoToStatus(errno);
  }
  return Status::OK;
}

Status HadoopFileSystem::IsDirectory(const std::string& fname) {
  hdfsFS fs = nullptr;
  FS_RETURN_IF_ERR(Connect(&fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr || info->mKind != kObjectKindDirectory) {
    PLOG(WARNING) << fname;
    return ErrnoToStatus(errno);
  }
  hdfs_->hdfsFreeFileInfo(info, 1);
  return Status::OK;
}

}  // namespace fs

}  // namespace oneflow
