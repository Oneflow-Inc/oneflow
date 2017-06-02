#ifndef ONEFLOW_CORE_RUNTIME_SNAPSHOT_H_
#define ONEFLOW_CORE_RUNTIME_SNAPSHOT_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class Snapshot final {
 public:
  class InStream;
  class OutStream;

  OF_DISALLOW_COPY_AND_MOVE(Snapshot);
  Snapshot() = delete;
  ~Snapshot() = default;
  
  Snapshot(const std::string& snapshot_root_path);

  // Get Stream
  std::unique_ptr<InStream> GetInStream(const std::string& key,
                                        size_t begin_pos);
  std::unique_ptr<OutStream> GetOutStream(const std::string& key,
                                          int32_t part_id);

 private:

};

class Snapshot::InStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InStream);
  InStream() = delete;
  ~InStream() = default;

  template<typename T>
  InStream& operator >> (T& x) {
    TODO();
    return *this;
  }

 private:
};

class Snapshot::OutStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OutStream);
  OutStream() = delete;
  ~OutStream() = default;

  template<typename T>
  OutStream& operator << (const T& x) {
    TODO();
    return *this;
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_CORE_RUNTIME_SNAPSHOT_H_
