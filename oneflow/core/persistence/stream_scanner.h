#ifndef ONEFLOW_CORE_PERSISTENCE_STREAM_SCANNER_H_
#define ONEFLOW_CORE_PERSISTENCE_STREAM_SCANNER_H_

#include <vector>
#include <string>
#include "oneflow/core/persistence/binary_in_stream.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class StreamScanner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamScanner);

  StreamScanner(fs::FileSystem* fs, const std::vector<std::string>& file_paths, uint64_t offset,
                bool with_local_copy);
  bool IsEof() const;
  uint64_t UpdateBuffer(std::vector<char>* buffer);

 protected:
  virtual void AddNForCurFilePos(uint64_t n) = 0;

  std::vector<std::unique_ptr<BinaryInStream>> streams_;
  uint64_t whole_file_size_;
  uint64_t whole_file_pos_;
  int32_t cur_stream_id_;
  int32_t stream_num_;

 private:
  uint64_t whole_file_offset_;
  bool with_local_copy_;

  void AddStream(fs::FileSystem* fs, const std::string& file_path, int64_t idx);
};

class CyclicStreamScanner final : public StreamScanner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CyclicStreamScanner);
  CyclicStreamScanner(fs::FileSystem* fs, const std::vector<std::string>& file_paths,
                      uint64_t offset, bool with_local_copy)
      : StreamScanner(fs, file_paths, offset, with_local_copy) {}

 protected:
  void AddNForCurFilePos(uint64_t n) override;
};

class AcyclicStreamScanner final : public StreamScanner {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AcyclicStreamScanner);
  AcyclicStreamScanner(fs::FileSystem* fs, const std::vector<std::string>& file_paths,
                       uint64_t offset, bool with_local_copy)
      : StreamScanner(fs, file_paths, offset, with_local_copy) {}

 protected:
  void AddNForCurFilePos(uint64_t n) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_STREAM_SCANNER_H_
