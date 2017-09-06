#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

TEST(Snapshot, write_and_read) {
  fs::FileSystem* file_system = fs::GetGlobalFileSystem();
  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string snapshot_root_path =
      JoinPath(current_dir, "/tmp_snapshot_test_asdfasdf");
  if (file_system->IsDirectory(snapshot_root_path) == fs::Status::OK) {
    std::vector<std::string> children;
    FS_CHECK_OK(file_system->GetChildren(snapshot_root_path, &children));
    ASSERT_EQ(children.size(), 0);
  } else {
    ASSERT_TRUE(file_system->CreateDir(snapshot_root_path) == fs::Status::OK);
  }
  std::string key = "key1";
  // write
  {
    Snapshot snapshot_write(snapshot_root_path);
    auto write_stream1_ptr = snapshot_write.GetOutStream(key, 0, 2);
    auto write_stream2_ptr = snapshot_write.GetOutStream(key, 1, 2);
    (*write_stream1_ptr) << 'a';
    snapshot_write.OnePartDone4Key(key, 0);
    (*write_stream2_ptr) << 'b';
    snapshot_write.OnePartDone4Key(key, 1);
  }
  // test write
  {
    std::string file1 = JoinPath(snapshot_root_path, key, "0");
    std::string file2 = JoinPath(snapshot_root_path, key, "1");
    std::unique_ptr<fs::RandomAccessFile> file1_ptr;
    std::unique_ptr<fs::RandomAccessFile> file2_ptr;
    ASSERT_TRUE(file_system->NewRandomAccessFile(file1, &file1_ptr)
                == fs::Status::OK);
    uint64_t file1_size = 0;
    ASSERT_TRUE(file_system->GetFileSize(file1, &file1_size) == fs::Status::OK);
    ASSERT_EQ(file1_size, 1);
    char* read_array1 = new char[file1_size];
    ASSERT_TRUE(file1_ptr->Read(0, file1_size, read_array1) == fs::Status::OK);
    std::string data1(read_array1, file1_size);
    ASSERT_TRUE(file_system->NewRandomAccessFile(file2, &file2_ptr)
                == fs::Status::OK);
    uint64_t file2_size = 0;
    ASSERT_TRUE(file_system->GetFileSize(file2, &file2_size) == fs::Status::OK);
    ASSERT_EQ(file2_size, 1);
    char* read_array2 = new char[file2_size];
    ASSERT_TRUE(file2_ptr->Read(0, file2_size, read_array2) == fs::Status::OK);
    std::string data2(read_array2, file2_size);
    ASSERT_EQ(data1, "a");
    ASSERT_EQ(data2, "b");
  }
  // read
  {
    Snapshot snapshot_read(snapshot_root_path);
    auto read_stream_ptr = snapshot_read.GetInStream(key, 0);
    char result;
    (*read_stream_ptr) >> result;
    ASSERT_EQ(result, 'a');
    (*read_stream_ptr) >> result;
    ASSERT_EQ(result, 'b');
    ASSERT_TRUE((*read_stream_ptr).good());
    (*read_stream_ptr) >> result;
    ASSERT_TRUE((*read_stream_ptr).eof());
  }
  {
    Snapshot snapshot_read(snapshot_root_path);
    auto read_stream_ptr = snapshot_read.GetInStream(key, 1);
    char result;
    (*read_stream_ptr) >> result;
    ASSERT_EQ(result, 'b');
    ASSERT_TRUE((*read_stream_ptr).good());
    (*read_stream_ptr) >> result;
    ASSERT_TRUE((*read_stream_ptr).eof());
  }
  int64_t undeletefiles, undeletedirs;
  ASSERT_TRUE(file_system->DeleteRecursively(snapshot_root_path, &undeletefiles,
                                             &undeletedirs)
              == fs::Status::OK);
}

}  // namespace oneflow
