//#include "gtest/gtest.h"
//#include "oneflow/core/common/process_state.h"
//#include "oneflow/core/common/str_util.h"
//#include "oneflow/core/persistence/posix/posix_file_system.h"
//#include "oneflow/core/persistence/windows/windows_file_system.h"
//
// namespace oneflow {
//
// namespace fs {
//
// void TestFileOperation(FileSystem* file_system) {
//  std::string current_dir = GetCwd();
//  StringReplace(&current_dir, '\\', '/');
//  std::string file_name = JoinPath(current_dir, "/tmp_test_file_asdfasdf");
//  // write
//  std::unique_ptr<WritableFile> writable_file;
//  ASSERT_TRUE(file_system->NewWritableFile(file_name, &writable_file)
//              == Status::OK);
//  std::string write_content = "oneflow-file-system-test";
//  ASSERT_TRUE(writable_file->Append(write_content.substr(0, 10).c_str(), 10)
//              == Status::OK);
//  ASSERT_TRUE(writable_file->Flush() == Status::OK);
//  ASSERT_TRUE(writable_file->Append(write_content.substr(10, 14).c_str(), 14)
//              == Status::OK);
//  ASSERT_TRUE(writable_file->Close() == Status::OK);
//  // write append
//  std::string append_content = "append-text";
//  std::unique_ptr<WritableFile> appendable_file;
//  ASSERT_TRUE(file_system->NewAppendableFile(file_name, &appendable_file)
//              == Status::OK);
//  ASSERT_TRUE(appendable_file->Append(append_content.c_str(), 11)
//              == Status::OK);
//  ASSERT_TRUE(appendable_file->Flush() == Status::OK);
//  ASSERT_TRUE(appendable_file->Close() == Status::OK);
//  // rename
//  std::string new_file_name = file_name + "_new";
//  ASSERT_TRUE(file_system->RenameFile(file_name, new_file_name) ==
//  Status::OK); ASSERT_TRUE(file_system->RenameFile(new_file_name, file_name)
//  == Status::OK);
//  // read
//  std::unique_ptr<RandomAccessFile> random_access_file;
//  ASSERT_TRUE(file_system->NewRandomAccessFile(file_name, &random_access_file)
//              == Status::OK);
//  uint64_t file_size = 0;
//  ASSERT_TRUE(file_system->GetFileSize(file_name, &file_size) == Status::OK);
//  ASSERT_EQ(file_size, 35);
//  char* read_array = new char[file_size];
//  ASSERT_TRUE(random_access_file->Read(0, file_size, read_array) ==
//  Status::OK); std::string read_content(read_array, file_size);
//  ASSERT_EQ(write_content + append_content, read_content);
//  ASSERT_TRUE(file_system->DeleteFile(file_name) == Status::OK);
//  delete[] read_array;
//}
//
// void TestDirOperation(FileSystem* file_system) {
//  std::string current_dir = GetCwd();
//  StringReplace(&current_dir, '\\', '/');
//  std::string test_root_path = JoinPath(current_dir,
//  "/tmp_test_dir_asdfasdf"); if (file_system->IsDirectory(test_root_path) ==
//  Status::OK) {
//    std::vector<std::string> children;
//    file_system->GetChildren(test_root_path, &children);
//    ASSERT_EQ(children.size(), 0);
//  } else {
//    ASSERT_TRUE(file_system->CreateDir(test_root_path) == Status::OK);
//  }
//  std::string file_name = JoinPath(test_root_path, "/direct_file_");
//  std::string content = "test_file";
//  std::unique_ptr<WritableFile> file_a;
//  std::unique_ptr<WritableFile> file_b;
//  ASSERT_TRUE(file_system->NewWritableFile(file_name + "_a", &file_a)
//              == Status::OK);
//  ASSERT_TRUE(file_a->Append(content.c_str(), 9) == Status::OK);
//  ASSERT_TRUE(file_a->Close() == Status::OK);
//  ASSERT_TRUE(file_system->NewWritableFile(file_name + "_b", &file_b)
//              == Status::OK);
//  ASSERT_TRUE(file_b->Append(content.c_str(), 9) == Status::OK);
//  ASSERT_TRUE(file_b->Close() == Status::OK);
//  std::string child_dir = JoinPath(test_root_path, "/direct_dir");
//  ASSERT_TRUE(file_system->CreateDir(child_dir) == Status::OK);
//  {
//    std::vector<std::string> children;
//    file_system->GetChildren(test_root_path, &children);
//    ASSERT_EQ(children.size(), 3);
//  }
//  int64_t undeleted_files = 0;
//  int64_t undeleted_dirs = 0;
//  ASSERT_TRUE(file_system->DeleteDir(child_dir) == Status::OK);
//  ASSERT_TRUE(file_system->IsDirectory(child_dir) != Status::OK);
//  ASSERT_TRUE(file_system->DeleteRecursively(test_root_path, &undeleted_files,
//                                             &undeleted_dirs)
//              == Status::OK);
//  ASSERT_EQ(undeleted_files, 0);
//  ASSERT_EQ(undeleted_dirs, 0);
//  ASSERT_TRUE(file_system->IsDirectory(test_root_path) != Status::OK);
//}
//
// void TestFileSystem(FileSystem* file_system) {
//  TestFileOperation(file_system);
//  TestDirOperation(file_system);
//}
//
//}  // namespace fs
//
// TEST(file_system, write_and_read) {
//#if defined(__linux__)
//  fs::FileSystem* file_system = new fs::PosixFileSystem();
//  fs::TestFileSystem(file_system);
//#elif defined(_WIN32)
//  fs::FileSystem* file_system = new fs::WindowsFileSystem();
//  fs::TestFileSystem(file_system);
//#endif
//}
//
//}  // namespace oneflow
