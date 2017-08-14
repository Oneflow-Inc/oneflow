#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

TEST(Snapshot, write_and_read) {
  tensorflow::Env* env = tensorflow::Env::Default();
  std::string current_dir = GetCwd();
  str_replace(&current_dir, '\\', '/');
  std::string snapshot_root_path =
      io::JoinPath(current_dir, "/tmp_snapshot_test_asdfasdf");
  if (env->IsDirectory(snapshot_root_path).code() == tensorflow::error::OK) {
    std::vector<std::string> children;
    TF_CHECK_OK(env->GetChildren(snapshot_root_path, &children));
    ASSERT_EQ(children.size(), 0);
  } else {
    ASSERT_TRUE(env->CreateDir(snapshot_root_path).code()
                == tensorflow::error::OK);
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
  std::string file1 = io::JoinPath(snapshot_root_path, key, "0");
  std::string file2 = io::JoinPath(snapshot_root_path, key, "1");
  std::string data1;
  std::string data2;
  TF_CHECK_OK(tensorflow::ReadFileToString(env, file1, &data1));
  TF_CHECK_OK(tensorflow::ReadFileToString(env, file2, &data2));
  ASSERT_EQ(data1, "a");
  ASSERT_EQ(data2, "b");
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
  tensorflow::int64 undeletefiles, undeletedirs;
  ASSERT_TRUE(
      env->DeleteRecursively(snapshot_root_path, &undeletefiles, &undeletedirs)
          .code()
      == tensorflow::error::OK);
}

}  // namespace oneflow
