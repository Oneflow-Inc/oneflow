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

#define FS_RETURN_FALSE_IF_FALSE(val) \
  if (!val) {                         \
    PLOG(WARNING);                    \
    return false;                     \
  }

namespace oneflow {

namespace fs {

namespace internal {

#ifdef PLATFORM_POSIX

bool GetSymbolFromLibrary(void* handle, const char* symbol_name,
                          void** symbol) {
  *symbol = dlsym(handle, symbol_name);
  if (!*symbol) {
    PLOG(WARNING) << dlerror();
    return false;
  }
  return true;
}

bool LoadLibrary(const char* library_filename, void** handle) {
  *handle = dlopen(library_filename, RTLD_NOW | RTLD_LOCAL);
  if (!*handle) {
    PLOG(WARNING) << dlerror();
    return false;
  }
  return true;
}

#endif  // PLATFORM_POSIX

#ifdef PLATFORM_WINDOWS

bool LoadLibrary(const char* library_filename, void** handle) {
  std::string file_name = library_filename;
  std::replace(file_name.begin(), file_name.end(), '/', '\\');

  std::wstring ws_file_name(WindowsFileSystem::Utf8ToWideChar(file_name));

  HMODULE hModule =
      LoadLibraryExW(ws_file_name.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!hModule) {
    PLOG(WARNING) << file_name + "not found";
    return false;
  }
  *handle = hModule;
  return true;
}

bool GetSymbolFromLibrary(void* handle, const char* symbol_name,
                          void** symbol) {
  FARPROC found_symbol;
  found_symbol = GetProcAddress((HMODULE)handle, symbol_name);
  if (found_symbol == NULL) {
    PLOG(WARNING) << std::string(symbol_name) + "not found";
    return false;
  }
  *symbol = (void**)found_symbol;
  return true;
}

#endif  // PLATFORM_WINDOWS

}  // namespace internal

template<typename R, typename... Args>
bool BindFunc(void* handle, const char* name, std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;
  FS_RETURN_FALSE_IF_FALSE(
      internal::GetSymbolFromLibrary(handle, name, &symbol_ptr));
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return true;
}

void LibHDFS::LoadAndBind() {
  auto TryLoadAndBind = [this](const char* name, void** handle) -> bool {
    FS_RETURN_FALSE_IF_FALSE(internal::LoadLibrary(name, handle));
#define BIND_HDFS_FUNC(function) \
  FS_RETURN_FALSE_IF_FALSE(BindFunc(*handle, #function, &function));

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
    return true;
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
    status_ = false;
    return;
  }
  std::string path = JoinPath(hdfs_home, "lib", "native", kLibHdfsDso);
  status_ = TryLoadAndBind(path.c_str(), &handle_);
  if (!status_) {
    // try load libhdfs.so using dynamic loader's search path in case
    // libhdfs.so is installed in non-standard location
    status_ = TryLoadAndBind(kLibHdfsDso, &handle_);
  }
}

HadoopFileSystem::HadoopFileSystem(const HdfsConf& hdfs_conf)
    : namenode_(hdfs_conf.namenode()), hdfs_(LibHDFS::Load()) {}

bool HadoopFileSystem::Connect(hdfsFS* fs) {
  FS_RETURN_FALSE_IF_FALSE(hdfs_->status());
  hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
  hdfs_->hdfsBuilderSetNameNode(builder, namenode_.c_str());
  // KERB_TICKET_CACHE_PATH will be deleted in the future, Because KRB5CCNAME
  // is the build in environment variable of Kerberos, so
  // KERB_TICKET_CACHE_PATH and related code are unnecessary.
  char* ticket_cache_path = getenv("KERB_TICKET_CACHE_PATH");
  if (ticket_cache_path != nullptr) {
    hdfs_->hdfsBuilderSetKerbTicketCachePath(builder, ticket_cache_path);
  }
  *fs = hdfs_->hdfsBuilderConnect(builder);
  if (*fs == nullptr) {
    PLOG(WARNING) << " HDFS connect failed. NOT FOUND";
    return false;
  }
  return true;
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

  void Read(uint64_t offset, size_t n, char* result) const override {
    char* dst = result;
    bool eof_retried = false;
    while (n > 0) {
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
        PCHECK(file_ == nullptr || hdfs_->hdfsCloseFile(fs_, file_) == 0)
            << filename_;
        file_ =
            hdfs_->hdfsOpenFile(fs_, hdfs_filename_.c_str(), O_RDONLY, 0, 0, 0);
        PCHECK(file_ != nullptr) << filename_;
        eof_retried = true;
      } else if (eof_retried && r == 0) {
        PLOG(FATAL) << "Read less bytes than requested";
        return;
      } else if (errno == EINTR || errno == EAGAIN) {
        // hdfsPread may return EINTR too. Just retry.
      } else {
        PLOG(FATAL) << filename_;
        return;
      }
    }
  }

 private:
  std::string filename_;
  std::string hdfs_filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;

  mutable std::mutex mu_;
  mutable hdfsFile file_;
};

void HadoopFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_RDONLY, 0, 0, 0);
  PCHECK(file != nullptr) << fname;
  result->reset(
      new HDFSRandomAccessFile(fname, TranslateName(fname), hdfs_, fs, file));
  CHECK_NOTNULL(result->get());
}

class HDFSWritableFile : public WritableFile {
 public:
  HDFSWritableFile(const std::string& fname, LibHDFS* hdfs, hdfsFS fs,
                   hdfsFile file)
      : filename_(fname), hdfs_(hdfs), fs_(fs), file_(file) {}

  ~HDFSWritableFile() override {
    if (file_ != nullptr) { Close(); }
  }

  void Append(const char* data, size_t n) override {
    PCHECK(hdfs_->hdfsWrite(fs_, file_, data, static_cast<tSize>(n)) != -1)
        << filename_;
  }

  void Close() override {
    int32_t result = hdfs_->hdfsCloseFile(fs_, file_);
    hdfs_ = nullptr;
    fs_ = nullptr;
    file_ = nullptr;
    PCHECK(result == 0) << filename_;
  }

  void Flush() override {
    PCHECK(hdfs_->hdfsHFlush(fs_, file_) == 0) << filename_;
  }

 private:
  std::string filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;
  hdfsFile file_;
};

void HadoopFileSystem::NewWritableFile(const std::string& fname,
                                       std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_WRONLY, 0, 0, 0);
  PCHECK(file != nullptr) << fname;
  result->reset(new HDFSWritableFile(fname, hdfs_, fs, file));
  CHECK_NOTNULL(result->get());
}

void HadoopFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  hdfsFile file = hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                                      O_WRONLY | O_APPEND, 0, 0, 0);
  PCHECK(file != nullptr) << fname;
  result->reset(new HDFSWritableFile(fname, hdfs_, fs, file));
  CHECK_NOTNULL(result->get());
}

bool HadoopFileSystem::FileExists(const std::string& fname) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));
  if (hdfs_->hdfsExists(fs, TranslateName(fname).c_str()) == 0) { return true; }
  return false;
}

std::vector<std::string> HadoopFileSystem::ListDir(const std::string& dir) {
  std::vector<std::string> result;
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
  // check to verify the directory exists first.
  CHECK(IsDirectory(dir));

  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info == nullptr) {
    // Assume it's an empty directory.
    return result;
  }
  for (int i = 0; i < entries; i++) {
    result.push_back(Basename(info[i].mName));
  }
  hdfs_->hdfsFreeFileInfo(info, entries);
  return result;
}

void HadoopFileSystem::DeleteFile(const std::string& fname) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));
  PCHECK(hdfs_->hdfsDelete(fs, TranslateName(fname).c_str(), /*recursive=*/0)
         == 0)
      << fname;
}

void HadoopFileSystem::CreateDir(const std::string& dir) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  PCHECK(hdfs_->hdfsCreateDirectory(fs, TranslateName(dir).c_str()) == 0)
      << dir;
}

void HadoopFileSystem::DeleteDir(const std::string& dir) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  // Count the number of entries in the directory, and only delete if it's
  // non-empty. This is consistent with the interface, but note that there's
  // a race condition where a file may be added after this check, in which
  // case the directory will still be deleted.
  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info != nullptr) { hdfs_->hdfsFreeFileInfo(info, entries); }
  // Due to HDFS bug HDFS-8407, we can't distinguish between an error and empty
  // folder, expscially for Kerberos enable setup, EAGAIN is quite common
  // when the call is actually successful. Check again by Stat.
  if (info == nullptr && errno != 0) { CHECK(IsDirectory(dir)); }
  PCHECK(entries == 0) << dir << "Cannot delete a non-empty directory.";
  PCHECK(hdfs_->hdfsDelete(fs, TranslateName(dir).c_str(), /*recursive=*/1)
         == 0)
      << dir;
}

uint64_t HadoopFileSystem::GetFileSize(const std::string& fname) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  PCHECK(info != nullptr) << fname;
  hdfs_->hdfsFreeFileInfo(info, 1);
  return static_cast<uint64_t>(info->mSize);
}

void HadoopFileSystem::RenameFile(const std::string& old_name,
                                  const std::string& new_name) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  PCHECK(
      hdfs_->hdfsExists(fs, TranslateName(new_name).c_str()) != 0
      || hdfs_->hdfsDelete(fs, TranslateName(new_name).c_str(), /*recursive=*/0)
             == 0)
      << new_name;

  PCHECK(hdfs_->hdfsRename(fs, TranslateName(old_name).c_str(),
                           TranslateName(new_name).c_str())
         == 0)
      << old_name;
}

bool HadoopFileSystem::IsDirectory(const std::string& fname) {
  hdfsFS fs = nullptr;
  CHECK(Connect(&fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr || info->mKind != kObjectKindDirectory) { return false; }
  hdfs_->hdfsFreeFileInfo(info, 1);
  return true;
}

}  // namespace fs

}  // namespace oneflow
