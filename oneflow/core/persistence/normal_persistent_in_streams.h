#ifndef ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAMS_H_
#define ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAMS_H_

#include <vector>
#include "oneflow/core/common/range.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/normal_persistent_in_stream.h"

namespace oneflow {

class NormalPersistentInStreams final : public PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalPersistentInStreams);
  NormalPersistentInStreams() = delete;

  NormalPersistentInStreams(fs::FileSystem* fs, const std::string& file_path_prefix, int32_t min_id,
                            int32_t max_id)
      : cur_stream_id_(0) {
    LOG(INFO) << "New NormalPersistentInStreams " << file_path_prefix << "-[" << min_id << ","
              << max_id << "]";
    CHECK_LE(min_id, max_id);
    FOR_RANGE(int32_t, part_id, min_id, max_id + 1) {
      std::string file_path = file_path_prefix + std::to_string(part_id);
      streams_.emplace_back(new NormalPersistentInStream(fs, file_path, 0));
    }
  }

  int32_t ReadLine(std::string* l) override;
  int32_t Read(char* s, size_t n) override;

 private:
  std::vector<std::unique_ptr<NormalPersistentInStream>> streams_;
  int32_t cur_stream_id_;

  int32_t DoRead(std::function<int32_t()> handler);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_NORMAL_PERSISTENT_IN_STREAMS_H_
