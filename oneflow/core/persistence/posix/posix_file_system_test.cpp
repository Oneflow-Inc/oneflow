#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {
  TEST(PosixFileSystem, write_and_read) {
    oneflow::PosixFileSystem* file_system = new oneflow::PosixFileSystem();
    std::string current_dir = GetCwd();
    StringReplace(&current_dir, '\\', '/');
    std::string test_root_path = 
      JoinPath(current_dir, "/tmp_posix_file_system_test_asdfasdf");
    if (file_system->IsDirectory(test_root_path) == oneflow::Status::OK) {
      std::vector<std::string> children;
      file_system->GetChildren(test_root_path, &children);
      ASSERT_EQ(childiren.size(), 0);
    } else {
      ASSERT_TRUE(file_system->CreateDir(test_root_path)
          == oneflow::Status::OK);
    }
    std::string file_name = 
    // write
    std::unique_ptr<WritableFile>* writable_file;
    ASSERT_TRUE(file_system->NewWritableFile(

    delete file_system;
  }
}  // namespace oneflow
