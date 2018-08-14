#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/common/process_state.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

TEST(Snapshot, write_and_read) {
  JobDescProto jb_desc_proto;
  auto job_conf = jb_desc_proto.mutable_job_conf();
  auto job_other = job_conf->mutable_other();
  auto persistence_path_conf = job_conf->mutable_other()->mutable_persistence_path_conf();
  persistence_path_conf->set_allocated_localfs_conf(new LocalFsConf);
  auto resource = jb_desc_proto.mutable_resource();
  resource->add_machine();
  Global<JobDesc>::Get()->InitFromProto(jb_desc_proto);
  fs::FileSystem* persistence_fs = GetFS(Global<JobDesc>::Get()->persistence_path_conf());

  std::string current_dir = GetCwd();
  StringReplace(&current_dir, '\\', '/');
  std::string snapshot_root_path = JoinPath(current_dir, "/tmp_snapshot_test_asdfasdf");
  if (persistence_fs->IsDirectory(snapshot_root_path)) {
    ASSERT_TRUE(persistence_fs->ListDir(snapshot_root_path).empty());
  } else {
    persistence_fs->CreateDir(snapshot_root_path);
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
    auto read_stream_ptr = std::make_unique<NormalPersistentInStream>(
        persistence_fs, JoinPath(snapshot_root_path, key));
    std::string content;
    read_stream_ptr->ReadLine(&content);
    ASSERT_EQ(content, "ab");
  }
  persistence_fs->RecursivelyDeleteDir(snapshot_root_path);
  ASSERT_TRUE(!persistence_fs->IsDirectory(snapshot_root_path));
}

}  // namespace oneflow
