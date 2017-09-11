#ifndef ONEFLOW_CORE_PERSISTENCE_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_FILE_SYSTEM_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/platform.h"

#if defined(_WIN32)
#undef DeleteFile
#endif

namespace oneflow {

namespace fs {

enum class Status {
  OK = 0,
  CANCELLED,
  UNKNOWN,
  INVALID_ARGUMENT,
  DEADLINE_EXCEEDED,
  NOT_FOUND,
  ALREADY_EXISTS,
  PERMISSION_DENIED,
  UNAUTHENTICATED,
  RESOURCE_EXHAUSTED,
  FAILED_PRECONDITION,
  ABORTED,
  OUT_OF_RANGE,
  UNIMPLEMENTED,
  INTERNAL,
  UNAVAILABLE,
  DATA_LOSS,
};

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(Status);

// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomAccessFile);
  RandomAccessFile() = default;
  virtual ~RandomAccessFile() = default;

  // Reads up to `n` bytes from the file starting at `offset`.
  //
  // Sets `*result` to the data that was read (including if fewer
  // than `n` bytes were successfully read).
  //
  // On OK returned status: `n` bytes have been stored in `*result`.
  // On non-OK returned status: `[0..n]` bytes have been stored in `*result`.
  //
  // Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  // because of EOF.
  //
  // Safe for concurrent use by multiple threads.
  virtual Status Read(uint64_t offset, size_t n, char* result) const = 0;

 private:
};

//  A file abstraction for sequential writing.
//
// The implementation must provide buffering since callers may append
// small fragments at a time to the file.
class WritableFile {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WritableFile);
  WritableFile() = default;
  virtual ~WritableFile() = default;

  // Append 'data' to the file.
  virtual Status Append(const char* data, size_t n) = 0;

  // Close the file.
  //
  // Flush() and de-allocate resources associated with this file
  //
  // Typical return codes (not guaranteed to be exhaustive):
  //  * OK
  //  * Other codes, as returned from Flush()
  virtual Status Close() = 0;

  //  Flushes the file and optionally syncs contents to filesystem.
  //
  // This should flush any local buffers whose contents have not been
  // delivered to the filesystem.
  //
  // If the process terminates after a successful flush, the contents
  // may still be persisted, since the underlying filesystem may
  // eventually flush the contents.  If the OS or machine crashes
  // after a successful flush, the contents may or may not be
  // persisted, depending on the implementation.
  virtual Status Flush() = 0;

 private:
};

class FileSystem {
 public:
  virtual ~FileSystem() = default;

  // Creates a brand new random access read-only file with the
  // specified name.
  //
  // On success, stores a pointer to the new file in
  // *result and returns OK.  On failure stores NULL in *result and
  // returns non-OK.  If the file does not exist, returns a non-OK
  // status.
  //
  // The returned file may be concurrently accessed by multiple threads.
  //
  // The ownership of the returned RandomAccessFile is passed to the caller
  // and the object should be deleted when is not used.
  virtual Status NewRandomAccessFile(
      const std::string& fname, std::unique_ptr<RandomAccessFile>* result) = 0;

  // Creates an object that writes to a new file with the specified
  // name.
  //
  // Deletes any existing file with the same name and creates a
  // new file.  On success, stores a pointer to the new file in
  // *result and returns OK.  On failure stores NULL in *result and
  // returns non-OK.
  //
  // The returned file will only be accessed by one thread at a time.
  //
  // The ownership of the returned WritableFile is passed to the caller
  // and the object should be deleted when is not used.
  virtual Status NewWritableFile(const std::string& fname,
                                 std::unique_ptr<WritableFile>* result) = 0;

  // Creates an object that either appends to an existing file, or
  // writes to a new file (if the file does not exist to begin with).
  //
  // On success, stores a pointer to the new file in *result and
  // returns OK.  On failure stores NULL in *result and returns
  // non-OK.
  //
  // The returned file will only be accessed by one thread at a time.
  //
  // The ownership of the returned WritableFile is passed to the caller
  // and the object should be deleted when is not used.
  virtual Status NewAppendableFile(const std::string& fname,
                                   std::unique_ptr<WritableFile>* result) = 0;

  // Returns OK if the named path exists and NOT_FOUND otherwise.
  virtual Status FileExists(const std::string& fname) = 0;

  // Returns true if all the listed files exist, false otherwise.
  // if status is not null, populate the vector with a detailed status
  // for each file.
  virtual bool FilesExist(const std::vector<std::string>& files,
                          std::vector<Status>* status);

  // Returns the immediate children in the given directory.
  //
  // The returned paths are relative to 'dir'.
  virtual Status GetChildren(const std::string& dir,
                             std::vector<std::string>* result) = 0;

  // Deletes the named file.
  virtual Status DeleteFile(const std::string& fname) = 0;

  // Creates the specified directory.
  // Typical return codes:
  //  * OK - successfully created the directory.
  //  * ALREADY_EXISTS - directory with name dirname already exists.
  //  * PERMISSION_DENIED - dirname is not writable.
  virtual Status CreateDir(const std::string& dirname) = 0;

  void CreateDirIfNotExist(const std::string& dirname);
  bool IsDirEmpty(const std::string& dirname);
  size_t GetChildrenNumOfDir(const std::string& dirname);

  // Creates the specified directory and all the necessary
  // subdirectories.
  // Typical return codes:
  //  * OK - successfully created the directory and sub directories, even if
  //         they were already created.
  //  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  virtual Status RecursivelyCreateDir(const std::string& dirname);

  // Deletes the specified directory.
  virtual Status DeleteDir(const std::string& dirname) = 0;

  // Deletes the specified directory and all subdirectories and files
  // underneath it. undeleted_files and undeleted_dirs stores the number of
  // files and directories that weren't deleted (unspecified if the return
  // status is not OK).
  // REQUIRES: undeleted_files, undeleted_dirs to be not null.
  // Typical return codes:
  //  * OK - dirname exists and we were able to delete everything underneath.
  //  * NOT_FOUND - dirname doesn't exist
  //  * PERMISSION_DENIED - dirname or some descendant is not writable
  //  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  //                    implemented
  virtual Status DeleteRecursively(const std::string& dirname);

  // Stores the size of `fname` in `*file_size`.
  virtual Status GetFileSize(const std::string& fname, uint64_t* file_size) = 0;

  // Overwrites the target if it exists.
  virtual Status RenameFile(const std::string& src,
                            const std::string& target) = 0;

  // Translate an URI to a filename for the FileSystem implementation.
  //
  // The implementation in this class cleans up the path, removing
  // duplicate /'s, resolving .. and . (more details in
  // str_util.h CleanPath).
  virtual std::string TranslateName(const std::string& name) const;

  // Returns whether the given path is a directory or not.
  //
  // Typical return codes (not guaranteed exhaustive):
  //  * OK - The path exists and is a directory.
  //  * FAILED_PRECONDITION - The path exists and is not a directory.
  //  * NOT_FOUND - The path entry does not exist.
  //  * PERMISSION_DENIED - Insufficient permissions.
  //  * UNIMPLEMENTED - The file factory doesn't support directories.
  virtual Status IsDirectory(const std::string& fname) = 0;

 protected:
  FileSystem() = default;
};

// If `current_status` is OK, stores `new_status` into `current_status`.
// If `current_status` is NOT OK, preserves the current status,
void TryUpdateStatus(Status* current_status, const Status& new_status);

Status ErrnoToStatus(int err_number);

#define FS_RETURN_IF_ERR(val)        \
  {                                  \
    const Status _ret_if_err = val;  \
    if (_ret_if_err != Status::OK) { \
      PLOG(WARNING);                 \
      return _ret_if_err;            \
    }                                \
  }

}  // namespace fs

// file system check status is ok
#define FS_CHECK_OK(val) CHECK_EQ(val, fs::Status::OK);

fs::FileSystem* LocalFS();
fs::FileSystem* GlobalFS();

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_FILE_SYSTEM_H_
