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
      JoinPath(current_dir, "/tmp_posix_file_system_test_asdfasdf");
  if (file_system->IsDirectory(test_root_path) == Status::OK) {
    std::vector<std::string> children;
    file_system->GetChildren(test_root_path, &children);
    ASSERT_EQ(children.size(), 0);
  } else {
    ASSERT_TRUE(file_system->CreateDir(test_root_path) == Status::OK);
  }
  std::string file_name = JoinPath(current_dir, "/tmp_test_file");
  // write
  std::unique_ptr<WritableFile>* writable_file;
}

}  // namespace fs
TEST(file_system, write_and_read) {
  fs::FileSystem* file_system = new fs::PosixFileSystem();
  fs::TestFileSystem(file_system);
}

}  // namespace oneflow
