#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_

#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class Snapshot final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Snapshot);
  Snapshot() = delete;
  ~Snapshot() = default;

  Snapshot(const std::string& snapshot_root_path);

  // Get Stream
  std::unique_ptr<PersistentInStream> GetInStream(const std::string& key,
                                                  size_t begin_pos);
  std::unique_ptr<PersistentOutStream> GetOutStream(const std::string& key,
                                                    int32_t part_id,
                                                    int32_t part_num);

  void OnePartDone4Key(const std::string& key);

 private:
  // check the sub_dir of snapshot_root_path and files of sub_dir is legal
  // and concat the sub parallel file of the key
  void CheckAndConcat();

  // a uniform file name, this file is concated from
  //   {part_0, part_1, ... part_n}
  static const char* concat_file_name_;
  static const char* done_file_name_;
  HashMap<std::string, int32_t> key2part_cnt_;
  std::string root_path_;
  tensorflow::Env* env_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
