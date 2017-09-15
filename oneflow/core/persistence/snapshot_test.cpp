#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

TEST(Snapshot, write_and_read) {
  JobDescProto jb_desc_proto;
  auto job_conf = jb_desc_proto.mutable_job_conf();
  auto gfs_conf = job_conf->mutable_global_fs_conf();
  gfs_conf->set_allocated_localfs_conf(new LocalFsConf);
  auto resource = jb_desc_proto.mutable_resource();
  resource->add_machine();
  JobDesc::Singleton()->InitFromProto(jb_desc_proto);

  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string snapshot_root_path =
      JoinPath(current_dir, "/tmp_snapshot_test_asdfasdf");
  if (GlobalFS()->IsDirectory(snapshot_root_path)) {
    ASSERT_TRUE(GlobalFS()->ListDir(snapshot_root_path).empty());
  } else {
    GlobalFS()->CreateDir(snapshot_root_path);
  }

  std::string key = "key/name";
  // write
  {
    Snapshot snapshot_write(snapshot_root_path);
    {
      auto write_stream1_ptr = snapshot_write.GetOutStream(key, 0);
      (*write_stream1_ptr) << 'a';
    }
    snapshot_write.OnePartDone(key, 0, 2);
    {
      auto write_stream2_ptr = snapshot_write.GetOutStream(key, 1);
      (*write_stream2_ptr) << 'b';
    }
    snapshot_write.OnePartDone(key, 1, 2);
  }
  // read
  {
    auto read_stream_ptr = of_make_unique<NormalPersistentInStream>(
        GlobalFS(), JoinPath(snapshot_root_path, key));
    std::string content;
    read_stream_ptr->ReadLine(&content);
    ASSERT_EQ(content, "ab");
  }
  GlobalFS()->RecursivelyDeleteDir(snapshot_root_path);
  ASSERT_TRUE(!GlobalFS()->IsDirectory(snapshot_root_path));
}

}  // namespace oneflow
