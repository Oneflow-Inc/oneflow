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
#include <gtest/gtest.h>
#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/posix/posix_file_system.h"

namespace oneflow {

namespace fs {

void TestFileOperation(FileSystem* file_system) {
  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string file_name = JoinPath(current_dir, "/tmp_test_file_asdfasdf");
  // write
  std::unique_ptr<WritableFile> writable_file;
  file_system->NewWritableFile(file_name, &writable_file);
  std::string write_content = "oneflow-file-system-test";
  writable_file->Append(write_content.substr(0, 10).c_str(), 10);
  writable_file->Flush();
  writable_file->Append(write_content.substr(10, 14).c_str(), 14);
  writable_file->Close();
  // write append
  std::string append_content = "append-text";
  std::unique_ptr<WritableFile> appendable_file;
  file_system->NewAppendableFile(file_name, &appendable_file);
  appendable_file->Append(append_content.c_str(), 11);
  appendable_file->Flush();
  appendable_file->Close();
  // rename
  std::string new_file_name = file_name + "_new";
  file_system->RenameFile(file_name, new_file_name);
  file_system->RenameFile(new_file_name, file_name);
  // read
  std::unique_ptr<RandomAccessFile> random_access_file;
  file_system->NewRandomAccessFile(file_name, &random_access_file);
  uint64_t file_size = file_system->GetFileSize(file_name);
  ASSERT_EQ(file_size, 35);
  char* read_array = new char[file_size];
  random_access_file->Read(0, file_size, read_array);
  std::string read_content(read_array, file_size);
  ASSERT_EQ(write_content + append_content, read_content);
  file_system->DelFile(file_name);
  delete[] read_array;
}

void TestDirOperation(FileSystem* file_system) {
  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string test_root_path = JoinPath(current_dir, "/tmp_test_dir_asdfasdf");
  if (file_system->IsDirectory(test_root_path)) {
    ASSERT_TRUE(file_system->ListDir(test_root_path).empty());
  } else {
    file_system->CreateDir(test_root_path);
  }
  std::string file_name = JoinPath(test_root_path, "/direct_file_");
  std::string content = "test_file";
  std::unique_ptr<WritableFile> file_a;
  std::unique_ptr<WritableFile> file_b;
  file_system->NewWritableFile(file_name + "_a", &file_a);
  file_a->Append(content.c_str(), 9);
  file_a->Close();
  file_system->NewWritableFile(file_name + "_b", &file_b);
  file_b->Append(content.c_str(), 9);
  file_b->Close();
  std::string child_dir = JoinPath(test_root_path, "/direct_dir");
  file_system->CreateDir(child_dir);
  ASSERT_EQ(file_system->ListDir(test_root_path).size(), 3);
  file_system->DeleteDir(child_dir);
  ASSERT_TRUE(!file_system->IsDirectory(child_dir));
  file_system->RecursivelyDeleteDir(test_root_path);
  ASSERT_TRUE(!file_system->IsDirectory(test_root_path));
}

void TestFileSystem(FileSystem* file_system) {
  TestFileOperation(file_system);
  TestDirOperation(file_system);
}

}  // namespace fs

TEST(file_system, write_and_read) {
#ifdef OF_PLATFORM_POSIX
  fs::FileSystem* file_system = new fs::PosixFileSystem();
  fs::TestFileSystem(file_system);
#endif
}

}  // namespace oneflow
