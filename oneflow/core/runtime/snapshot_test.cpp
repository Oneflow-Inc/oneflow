#include "oneflow/core/runtime/snapshot.h"
#include "gtest/gtest.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/stream_executor/lib/process_state.h"

namespace oneflow {

// std::string BaseDir() { return "file://"; }

TEST(Snapshot, write_and_read) {
  tensorflow::Env* env = tensorflow::Env::Default();
  std::string current_dir;
  // perftools::gputools::port::GetCurrentDirectory(&current_dir);
  std::string snapshot_root_path = tensorflow::io::JoinPath("file://", "/C:/Users/oneflow/root_path/" ,"snapshot1");
  if (env->IsDirectory(snapshot_root_path).code() == tensorflow::error::OK) {
    std::vector<std::string> children;
    env->GetChildren(snapshot_root_path, &children);
    ASSERT_EQ(children.size(), 0);
  } else {
    ASSERT_TRUE(env->CreateDir(snapshot_root_path).code() == tensorflow::error::OK);
  }
  // write
  {
    Snapshot snapshot_write(snapshot_root_path);
    std::string key = "key1";
    auto write_stream1_ptr = snapshot_write.GetOutStream(key, 0);
    auto write_stream2_ptr = snapshot_write.GetOutStream(key, 1);
    (*write_stream1_ptr) << 'a';
    (*write_stream2_ptr) << 'b';
    std::string file1 = tensorflow::io::JoinPath(snapshot_root_path, key, "0");
    std::string file2 = tensorflow::io::JoinPath(snapshot_root_path, key, "1");
    std::string data1;
    std::string data2;
    tensorflow::ReadFileToString(env, file1, &data1);
    tensorflow::ReadFileToString(env, file2, &data2);
    ASSERT_EQ(data1, "a");
    ASSERT_EQ(data2, "b");
  }
  // read
  {
    Snapshot snapshot_read(snapshot_root_path);
    std::string key = "key1";
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
}

}  // namespace oneflow

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
