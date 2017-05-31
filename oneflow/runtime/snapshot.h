#ifndef ONEFLOW_RUNTIME_SNAPSHOT_H_
#define ONEFLOW_RUNTIME_SNAPSHOT_H_

#include "common/util.h"

namespace oneflow {

class Snapshot final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Snapshot);
  Snapshot() = delete;
  ~Snapshot() = default;

  void Read(const std::string& key, size_t begin_pos);
  
  void PrepareWrite(const std::string& key, int32_t part_num);
  void Write(const std::string& key, int32_t part_id);

 private:

};

} // namespace oneflow

#endif // ONEFLOW_RUNTIME_SNAPSHOT_H_
