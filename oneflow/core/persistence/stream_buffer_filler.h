#ifndef ONEFLOW_CORE_PERSISTENCE_STREAM_BUFFER_FILLER_H_
#define ONEFLOW_CORE_PERSISTENCE_STREAM_BUFFER_FILLER_H_

#include <vector>
#include "oneflow/core/persistence/binary_in_stream.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class StreamBufferFiller {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamBufferFiller);

  StreamBufferFiller(fs::FileSystem* fs, const std::string& file_path_prefix, int32_t min_id,
                     int32_t max_id, bool cyclic, bool with_local_copy);
  StreamBufferFiller(fs::FileSystem* fs, const std::string& file_path, uint64_t offset, bool cyclic,
                     bool with_local_copy);
  bool IsEof() const;
  uint64_t UpdateBuffer(std::vector<char>* buffer);

 private:
  std::vector<std::unique_ptr<BinaryInStream>> streams_;
  int32_t cur_stream_id_;
  int32_t stream_num_;

  uint64_t whole_file_size_;
  uint64_t whole_file_pos_;
  bool cyclic_;
  bool with_local_copy_;

  void AddNForCurFilePosNonCyclic(uint64_t n);
  void AddNForCurFilePosCyclic(uint64_t n);
  void AddStream(fs::FileSystem* fs, const std::string& file_path, uint64_t offset);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_STREAM_BUFFER_FILLER_H_
