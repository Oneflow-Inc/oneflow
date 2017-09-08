#ifndef ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
#define ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_

#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class Snapshot final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Snapshot);
  Snapshot() = delete;
  ~Snapshot() = default;

  Snapshot(const std::string& snapshot_root_path);

  std::unique_ptr<NormalPersistentInStream> GetInStream(const std::string& lbn,
                                                        size_t begin_pos) const;
  std::unique_ptr<NormalPersistentInStream> GetInStream(
      const std::string& lbn, int32_t part_id, int32_t part_num,
      int32_t dim_num, int64_t byte_size_of_each_dim) const;

  std::unique_ptr<PersistentOutStream> GetOutStream(const std::string& lbn,
                                                    int32_t part_id);

  void OnePartDone(const std::string& lbn, int32_t part_id, int32_t part_num);

 private:
  void ConcatLbnFile(const std::string& lbn, int32_t part_num,
                     const std::string& concat_file);

  std::string root_path_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_SNAPSHOT_H_
