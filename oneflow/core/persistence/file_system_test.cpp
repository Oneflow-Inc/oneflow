#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/persistence/windows/windows_file_system.h"

namespace oneflow {

namespace fs {

void TestFileSystem(FileSystem* file_system) {
  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string test_root_path =
      JoinPath(current_dir, "/tmp_file_system_test_asdfasdf");
  if (file_system->IsDirectory(test_root_path) == Status::OK) {
    std::vector<std::string> children;
    file_system->GetChildren(test_root_path, &children);
    ASSERT_EQ(children.size(), 0);
  } else {
    ASSERT_TRUE(file_system->CreateDir(test_root_path) == Status::OK);
  }
  std::string file_name = JoinPath(current_dir, "/tmp_test_file");
  // write
  std::unique_ptr<WritableFile> writable_file;
  ASSERT_TRUE(file_system->NewWritableFile(file_name, &writable_file)
              == Status::OK);
  std::string write_content = "oneflow-file-system-test";
  ASSERT_TRUE(writable_file->Append(write_content.substr(0, 10).c_str(), 10)
              == Status::OK);
  ASSERT_TRUE(writable_file->Flush() == Status::OK);
  ASSERT_TRUE(writable_file->Append(write_content.substr(10, 14).c_str(), 14)
              == Status::OK);
  ASSERT_TRUE(writable_file->Close() == Status::OK);
  // rename
  std::string new_file_name = file_name + "_new";
  ASSERT_TRUE(file_system->RenameFile(file_name, new_file_name) == Status::OK);
  ASSERT_TRUE(file_system->RenameFile(new_file_name, file_name) == Status::OK);
  // read
  std::unique_ptr<RandomAccessFile> random_access_file;
  ASSERT_TRUE(file_system->NewRandomAccessFile(file_name, &random_access_file)
              == Status::OK);
  uint64_t file_size = 0;
  ASSERT_TRUE(file_system->GetFileSize(file_name, &file_size) == Status::OK);
  ASSERT_EQ(file_size, 24);
  char* read_array = new char[file_size];
  ASSERT_TRUE(random_access_file->Read(0, file_size, read_array) == Status::OK);
  std::string read_content(read_array, file_size);
  ASSERT_EQ(write_content, read_content);
  ASSERT_TRUE(file_system->DeleteFile(file_name) == Status::OK);
  ASSERT_TRUE(file_system->DeleteDir(test_root_path) == Status::OK);
  ASSERT_TRUE(file_system->IsDirectory(test_root_path) != Status::OK);
  delete[] read_array;
}

}  // namespace fs

TEST(file_system, write_and_read) {
#if defined(__linux__)
  fs::FileSystem* file_system = new fs::PosixFileSystem();
  fs::TestFileSystem(file_system);
#elif defined(_WIN32)
  fs::FileSystem* file_system = new fs::WindowsFileSystem();
  fs::TestFileSystem(file_system);
#endif
}

}  // namespace oneflow
