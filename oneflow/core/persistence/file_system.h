/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_PERSISTENCE_FILE_SYSTEM_H_
#define ONEFLOW_CORE_PERSISTENCE_FILE_SYSTEM_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/file_system_conf.pb.h"

namespace oneflow {

namespace fs {

// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomAccessFile);
  RandomAccessFile() = default;
  virtual ~RandomAccessFile() = default;

  // Reads `n` bytes from the file starting at `offset`.
  // Sets `*result` to the data that was read.
  //
  // Safe for concurrent use by multiple threads.
  virtual void Read(uint64_t offset, size_t n, char* result) const = 0;

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
  virtual void Append(const char* data, size_t n) = 0;

  // Close the file.
  //
  // Flush() and de-allocate resources associated with this file
  virtual void Close() = 0;

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
  virtual void Flush() = 0;

 private:
};

class FileSystem {
 public:
  virtual ~FileSystem() = default;

  // Creates a brand new random access read-only file with the
  // specified name.
  //
  // On success, stores a pointer to the new file in
  // *result.  On failure stores NULL in *result.
  //
  // The returned file may be concurrently accessed by multiple threads.
  //
  // The ownership of the returned RandomAccessFile is passed to the caller
  // and the object should be deleted when is not used.
  virtual void NewRandomAccessFile(const std::string& fname,
                                   std::unique_ptr<RandomAccessFile>* result) = 0;

  // Creates an object that writes to a new file with the specified
  // name.
  //
  // Deletes any existing file with the same name and creates a
  // new file.  On success, stores a pointer to the new file in
  // *result.  On failure stores NULL in *result.
  //
  // The returned file will only be accessed by one thread at a time.
  //
  // The ownership of the returned WritableFile is passed to the caller
  // and the object should be deleted when is not used.
  virtual void NewWritableFile(const std::string& fname, std::unique_ptr<WritableFile>* result) = 0;

  // Creates an object that either appends to an existing file, or
  // writes to a new file (if the file does not exist to begin with).
  //
  // On success, stores a pointer to the new file in *result.
  // On failure stores NULL in *result.
  //
  // The returned file will only be accessed by one thread at a time.
  //
  // The ownership of the returned WritableFile is passed to the caller
  // and the object should be deleted when is not used.
  virtual void NewAppendableFile(const std::string& fname,
                                 std::unique_ptr<WritableFile>* result) = 0;

  // Returns true if the named path exists and false otherwise.
  virtual bool FileExists(const std::string& fname) = 0;

  // Returns the immediate children in the `dir`
  //
  // The returned paths are relative to 'dir'.
  virtual std::vector<std::string> ListDir(const std::string& dir) = 0;

  // Deletes the named file.
  // Using DelFile to avoid Windows macro
  virtual void DelFile(const std::string& fname) = 0;

  // Creates the specified directory.
  virtual void CreateDir(const std::string& dirname) = 0;
  void CreateDirIfNotExist(const std::string& dirname);
  virtual void RecursivelyCreateDir(const std::string& dirname);
  void RecursivelyCreateDirIfNotExist(const std::string& dirname);

  // Empty
  bool IsDirEmpty(const std::string& dirname);
  void MakeEmptyDir(const std::string& dirname);

  // Deletes the specified directory.
  virtual void DeleteDir(const std::string& dirname) = 0;

  // Deletes the specified directory and all subdirectories and files
  // underneath it. undeleted_files and undeleted_dirs stores the number of
  // files and directories that weren't deleted.
  virtual void RecursivelyDeleteDir(const std::string& dirname);

  // Returns the size of `fname`.
  virtual uint64_t GetFileSize(const std::string& fname) = 0;

  // Overwrites the target if it exists.
  virtual void RenameFile(const std::string& old_name, const std::string& new_name) = 0;

  // Translate an URI to a filename for the FileSystem implementation.
  //
  // The implementation in this class cleans up the path, removing
  // duplicate /'s, resolving .. and . (more details in
  // str_util.h CleanPath).
  virtual std::string TranslateName(const std::string& name) const;

  // Returns whether the given path is a directory or not.
  virtual bool IsDirectory(const std::string& fname) = 0;

 protected:
  FileSystem() = default;
};

}  // namespace fs

fs::FileSystem* LocalFS();

fs::FileSystem* GetFS(const FileSystemConf& file_system_conf);
fs::FileSystem* DataFS();
fs::FileSystem* DataFS(int64_t session_id);
fs::FileSystem* SnapshotFS();
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_FILE_SYSTEM_H_
