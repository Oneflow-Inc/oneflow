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

  void Read(uint64_t offset, size_t n, char* result) const override {
    char* dst = result;
    while (n > 0) {
      SSIZE_T r = pread(hfile_, dst, n, offset);
      if (r > 0) {
        offset += r;
        dst += r;
        n -= r;
      } else if (r == 0) {
        PLOG(FATAL) << "Read file warning for : " + filename_
                           + " Read fewer bytes than requested";
        return;
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        PLOG(FATAL) << "Read file warning for : " + filename_;
        return;
      }
    }
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

  void Append(const char* data, size_t n) override {
    DWORD bytes_written = 0;
    DWORD data_size = static_cast<DWORD>(n);
    BOOL write_result =
        ::WriteFile(hfile_, data, data_size, &bytes_written, NULL);
    PCHECK(FALSE != write_result) << "Failed to WriteFile: " + filename_;
    PCHECK(size_t(bytes_written) == n);
  }

  void Close() override {
    assert(INVALID_HANDLE_VALUE != hfile_);
    Flush();
    PCHECK(FALSE != ::CloseHandle(hfile_))
        << "CloseHandle failed for:" + filename_;
    hfile_ = INVALID_HANDLE_VALUE;
  }

  void Flush() override {
    PCHECK(FALSE != ::FlushFileBuffers(hfile_))
        << "FlushFileBuffers failed for: " + filename_;
  }
};

}  // namespace

void WindowsFileSystem::NewRandomAccessFile(
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
  PCHECK(INVALID_HANDLE_VALUE != hfile)
      << "NewRandomAccessFile failed to Create/Open: " + fname;
  result->reset(new WindowsRandomAccessFile(translated_fname, hfile));
}

void WindowsFileSystem::NewWritableFile(const std::string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  PCHECK(INVALID_HANDLE_VALUE != hfile)
      << "Failed to create a NewWriteableFile: " + fname;
  result->reset(new WindowsWritableFile(translated_fname, hfile));
}

void WindowsFileSystem::NewAppendableFile(
    const std::string& fname, std::unique_ptr<WritableFile>* result) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_fname = Utf8ToWideChar(translated_fname);
  result->reset();

  DWORD share_mode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  HANDLE hfile =
      ::CreateFileW(ws_translated_fname.c_str(), GENERIC_WRITE, share_mode,
                    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  PCHECK(INVALID_HANDLE_VALUE != hfile)
      << "Failed to create a NewAppendableFile: " + fname;
  UniqueCloseHandlePtr file_guard(hfile, CloseHandleFunc);
  DWORD file_ptr = ::SetFilePointer(hfile, NULL, NULL, FILE_END);
  PCHECK(INVALID_SET_FILE_POINTER != file_ptr)
      << "Failed to create a NewAppendableFile: " + fname;
  result->reset(new WindowsWritableFile(translated_fname, hfile));
  file_guard.release();
}

bool WindowsFileSystem::FileExists(const std::string& fname) {
  constexpr int kOk = 0;
  if (_access(TranslateName(fname).c_str(), kOk) == 0) { return true; }
  return false;
}

std::vector<std::string> WindowsFileSystem::ListDir(const std::string& dir) {
  std::string translated_dir = TranslateName(dir);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_dir);
  std::vector<std::string> result;

  std::wstring pattern = ws_translated_dir;
  if (!pattern.empty() && pattern.back() != '\\' && pattern.back() != '/') {
    pattern += L"\\*";
  } else {
    pattern += L'*';
  }

  WIN32_FIND_DATAW find_data;
  HANDLE find_handle = ::FindFirstFileW(pattern.c_str(), &find_data);
  PCHECK(find_handle != INVALID_HANDLE_VALUE)
      << "FindFirstFile failed for: " + translated_dir;
  do {
    std::string file_name = WideCharToUtf8(find_data.cFileName);
    if (file_name != "." && file_name != "..") { result.push_back(file_name); }
  } while (::FindNextFileW(find_handle, &find_data));
  PCHECK(::FindClose(find_handle)) << "FindClose failed for: " + translated_dir;
  return result;
}

void WindowsFileSystem::DelFile(const std::string& fname) {
  std::wstring file_name = Utf8ToWideChar(fname);
  PCHECK(_wunlink(file_name.c_str()) == 0)
      << "Failed to delete a file: " + fname;
}

void WindowsFileSystem::CreateDir(const std::string& name) {
  std::wstring ws_name = Utf8ToWideChar(name);
  PCHECK(_wmkdir(ws_name.c_str()) == 0)
      << "Failed to create a directory: " + name;
}

void WindowsFileSystem::DeleteDir(const std::string& name) {
  std::wstring ws_name = Utf8ToWideChar(name);
  PCHECK(_wrmdir(ws_name.c_str()) == 0)
      << "Failed to remove a directory: " + name;
}

uint64_t WindowsFileSystem::GetFileSize(const std::string& fname) {
  std::string translated_fname = TranslateName(fname);
  std::wstring ws_translated_dir = Utf8ToWideChar(translated_fname);
  WIN32_FILE_ATTRIBUTE_DATA attrs;
  PCHECK(TRUE
         == ::GetFileAttributesExW(ws_translated_dir.c_str(),
                                   GetFileExInfoStandard, &attrs))
      << "Can not get size for: " + fname;
  ULARGE_INTEGER file_size;
  file_size.HighPart = attrs.nFileSizeHigh;
  file_size.LowPart = attrs.nFileSizeLow;
  return file_size.QuadPart;
}

void WindowsFileSystem::RenameFile(const std::string& old_name,
                                   const std::string& new_name) {
  // rename() is not capable of replacing the existing file as on Linux
  // so use OS API directly
  std::wstring ws_translated_src = Utf8ToWideChar(TranslateName(old_name));
  std::wstring ws_translated_target = Utf8ToWideChar(TranslateName(new_name));
  PCHECK(::MoveFileExW(ws_translated_src.c_str(), ws_translated_target.c_str(),
                       MOVEFILE_REPLACE_EXISTING))
      << "Failed to rename: " + old_name + " to: " + new_name;
}

bool WindowsFileSystem::IsDirectory(const std::string& fname) {
  struct _stat sbuf;
  std::wstring ws_translated_fname = Utf8ToWideChar(TranslateName(fname));
  if (_wstat(ws_translated_fname.c_str(), &sbuf) != 0) {
    return false;
  } else if (PathIsDirectoryW(ws_translated_fname.c_str())) {
    return true;
  } else {
    return false;
  }
}

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS
