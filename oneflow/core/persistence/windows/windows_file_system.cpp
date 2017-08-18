#include "oneflow/core/persistence/windows/windows_file_system.h"
#include <Windows.h>

#ifdef PLATFORM_WINDOWS

#undef DeleteFile

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
    Status s;
    char* dst = result;
    while (n > 0 && s == Status::OK) {
      SSIZE_T r = pread(hfile_, dst, n, offset);
      if (r > 0) {
        offset += r;
        dst += r;
        n -= r;
      } else if (r == 0) {
        s = Status::OUT_OF_RANGE;
        PLOG(WARNING) << "Read file warning for : " + filename_
                             + " Read fewer bytes than requested";
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = ErrnoToStatus(errno);
        PLOG(WARNING) << "Read file warning for : " + filename_;
      }
    }
    if (s != Status::OK) {}
    return s;
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

}  // namespace fs

}  // namespace oneflow

#endif  // PLATFORM_WINDOWS
