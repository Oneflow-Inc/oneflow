#include "oneflow/core/persistence/windows/windows_file_system.h"

#ifdef PLATFORM_WINDOWS

#include <Shlwapi.h>

namespace oneflow {

namespace fs {

namespace {

// RAII helpers for HANDLEs
const auto CloseHandleFunc = [](HANDLE h) { ::CloseHandle(h); };
typedef std::unique_ptr<void, decltype(CloseHandleFunc)> UniqueCloseHandlePtr;

// PLEASE NOTE: hfile is expected to be an async handle
// (i.e. opened with FILE_FLAG_OVERLAPPED)
SSIZE_T pread(HANDLE hfile, char* src, size_t num_bytes, uint64_t offset) {
  assert(num_bytes <= std::numeric_limits<DWORD>::max());
  OVERLAPPED overlapped = {0};
  ULARGE_INTEGER offset_union;
  offset_union.QuadPart = offset;

  overlapped.Offset = offset_union.LowPart;
  overlapped.OffsetHigh = offset_union.HighPart;
  overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);

  if (NULL == overlapped.hEvent) { return -1; }

  SSIZE_T result = 0;

  unsigned long bytes_read = 0;
  DWORD last_error = ERROR_SUCCESS;

  BOOL read_result = ::ReadFile(hfile, src, static_cast<DWORD>(num_bytes),
                                &bytes_read, &overlapped);
  if (TRUE == read_result) {
    result = bytes_read;
  } else if ((FALSE == read_result)
             && ((last_error = GetLastError()) != ERROR_IO_PENDING)) {
    result = (last_error == ERROR_HANDLE_EOF) ? 0 : -1;
  } else {
    if (ERROR_IO_PENDING
        == last_error) {  // Otherwise bytes_read already has the result.
      BOOL overlapped_result =
          ::GetOverlappedResult(hfile, &overlapped, &bytes_read, TRUE);
      if (FALSE == overlapped_result) {
        result = (::GetLastError() == ERROR_HANDLE_EOF) ? 0 : -1;
      } else {
        result = bytes_read;
      }
    }
  }

  ::CloseHandle(overlapped.hEvent);

  return result;
}

// read() based random-access
class WindowsRandomAccessFile : public RandomAccessFile {
 private:
  std::string filename_;
  HANDLE hfile_;

 public:
  WindowsRandomAccessFile(const std::string& fname, HANDLE hfile)
      : filename_(fname), hfile_(hfile) {}
  ~WindowsRandomAccessFile() override {
    if (hfile_ != NULL && hfile_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(hfile_);
    }
  }

  Status Read(uint64_t offset, size_t n, char* result) const override {
    char* dst = result;
    while (n > 0) {
      SSIZE_T r = pread(hfile_, dst, n, offset);
      if (r > 0) {
        offset += r;
        dst += r;
        n -= r;
      } else if (r == 0) {
        PLOG(WARNING) << "Read file warning for : " + filename_
                             + " Read fewer bytes than requested";
        return Status::OUT_OF_RANGE;
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        PLOG(WARNING) << "Read file warning for : " + filename_;
        return ErrnoToStatus(errno);
      }
    }
    return Status::OK;
  }
};

class WindowsWritableFile : public WritableFile {
 private:
  std::string filename_;
  HANDLE hfile_;

 public:
  WindowsWritableFile(const std::string& fname, HANDLE hFile)
      : filename_(fname), hfile_(hFile) {}

  ~WindowsWritableFile() override {
    if (hfile_ != NULL && hfile_ != INVALID_HANDLE_VALUE) {
      WindowsWritableFile::Close();
    }
  }

  Status Append(const char* data, size_t n) override {
    DWORD bytes_written = 0;
    DWORD data_size = static_cast<DWORD>(n);
    BOOL write_result =
        ::WriteFile(hfile_, data, data_size, &bytes_written, NULL);
    if (FALSE == write_result) {
      PLOG(WARNING) << "Failed to WriteFile: " + filename_;
      return ErrnoToStatus(::GetLastError());
    }
    assert(size_t(bytes_written) == n);
    return Status::OK;
  }

  Status Close() override {
    assert(INVALID_HANDLE_VALUE != hfile_);
    Status result = Flush();
    if (result != Status::OK) { return result; }

    if (FALSE == ::CloseHandle(hfile_)) {
      PLOG(WARNING) << "CloseHandle failed for:" + filename_;
      return ErrnoToStatus(::GetLastError());
    }

    hfile_ = INVALID_HANDLE_VALUE;
    return Status::OK;
  }

  Status Flush() override {
    if (FALSE == ::FlushFileBuffers(hfile_)) {
      LOG(WARNING) << "FlushFileBuffers failed for: " + filename_;
      return ErrnoToStatus(::GetLastError());
    }
    return Status::OK;
  }
};

}  // namespace

Status WindowsFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  // Open the file for read-only random access
  // Open in async mode which makes Windows allow more parallelism even
  // if we need to do sync I/O on top of it.
  DWORD file_flags = FILE_ATTRIBUTE_READONLY | FILE_FLAG_OVERLAPPED;
  // Shared access is necessary for tests to pass
  // almost all tests would work with a possible exception of fault_injection.
  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;

  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_READ, share_mode, NULL,
                    OPEN_EXISTING, file_flags, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    std::string context = "NewRandomAccessFile failed to Create/Open: " + fname;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }

  result->reset(new WindowsRandomAccessFile(translated_fname, hfile));
  return Status::OK;
}

Status WindowsFileSystem::NewWritableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    std::string context = "Failed to create a NewWriteableFile: " + fname;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }

  result->reset(new WindowsWritableFile(translated_fname, hfile));
  return Status::OK;
}

Status WindowsFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

  if (INVALID_HANDLE_VALUE == hfile) {
    std::string context = "Failed to create a NewAppendableFile: " + fname;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }

  UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);

  DWORD file_ptr = ::SetFilePointer(hfile, NULL, NULL, FILE_END);
  if (INVALID_SET_FILE_POINTER == file_ptr) {
    std::string context = "Failed to create a NewAppendableFile: " + fname;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }

  result->reset(new WindowsWritableFile(translated_fname, hfile));
  file_guard.release();

  return Status::OK;
}

Status WindowsFileSystem::FileExists(const std::string& fname) {
  constexpr int kOk = 0;
  if (_access(TranslateName(fname).c_str(), kOk) == 0) { return Status::OK; }
  LOG(WARNING) << fname << " not found";
  return Status::NOT_FOUND;
}

Status WindowsFileSystem::GetChildren(const std::string& dir,
                                      std::vector<std::string>* result) {
  std::string translated_dir = TranslateName(dir);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_dir);
  result->clear();

  std::wstring pattern = ws_translated_dir;
  if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/') {
    pattern += L"\\*";
  } else {
    pattern += L'*';
  }

  WIN32_FIND_DATAW find_data;
  HANDLE find_handle = ::FindFirstFileW(pattern.c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    std::string context = "FindFirstFile failed for: " + translated_dir;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }

  do {
    std::string file_name = WideCharToUtf8(find_data.cFileName);
    if (file_name != "." && file_name != "..") { result->push_back(file_name); }
  } while (::FindNextFileW(find_handle, &find_data));

  if (!::FindClose(find_handle)) {
    std::string context = "FindClose failed for: " + translated_dir;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }

  return Status::OK;
}

Status WindowsFileSystem::DeleteFile(const std::string& fname) {
  std::wstring file_name = Utf8ToWideChar(fname);
  if (_wunlink(file_name.c_str()) != 0) {
    std::string context = "Failed to delete a file: " + fname;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }
  return Status::OK;
}

Status WindowsFileSystem::CreateDir(const std::string& name) {
  std::wstring ws_name = Utf8ToWideChar(name);
  if (_wmkdir(ws_name.c_str()) != 0) {
    std::string context = "Failed to create a directory: " + name;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }
  return Status::OK;
}

Status WindowsFileSystem::DeleteDir(const std::string& name) {
  std::wstring ws_name = Utf8ToWideChar(name);
  if (_wrmdir(ws_name.c_str()) != 0) {
    std::string context = "Failed to remove a directory: " + name;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }
  return Status::OK;
}

Status WindowsFileSystem::GetFileSize(const std::string& fname,
                                      uint64_t* size) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_fname);
  WIN32_FILE_ATTRIBUTE_DATA attrs;
  if (TRUE
      == ::GetFileAttributesExW(ws_translated_dir.c_str(),
                                GetFileExInfoStandard, &attrs)) {
    ULARGE_INTEGER file_size;
    file_size.HighPart = attrs.nFileSizeHigh;
    file_size.LowPart = attrs.nFileSizeLow;
    *size = file_size.QuadPart;
  } else {
    std::string context = "Can not get size for: " + fname;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }
  return Status::OK;
}

Status WindowsFileSystem::RenameFile(const std::string& src,
                                     const std::string& target) {
  // rename() is not capable of replacing the existing file as on Linux
  // so use OS API directly
  std::wstring ws_translated_src = Utf8ToWideChar(TranslateName(src));
  std::wstring ws_translated_target = Utf8ToWideChar(TranslateName(target));
  if (!::MoveFileExW(ws_translated_src.c_str(), ws_translated_target.c_str(),
                     MOVEFILE_REPLACE_EXISTING)) {
    std::string context = "Failed to rename: " + src + " to: " + target;
    LOG(WARNING) << context;
    return ErrnoToStatus(::GetLastError());
  }
  return Status::OK;
}

Status WindowsFileSystem::IsDirectory(const std::string& fname) {
  struct _stat sbuf;
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (_wstat(ws_translated_fname.c_str(), &sbuf) != 0) {
    LOG(WARNING) << fname;
    return ErrnoToStatus(::GetLastError());
  } else if (PathIsDirectoryW(ws_translated_fname.c_str())) {
    return Status::OK;
  } else {
    std::string context = fname + " not a directory";
    LOG(WARNING) << context;
    return Status::FAILED_PRECONDITION;
  }
}

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS
