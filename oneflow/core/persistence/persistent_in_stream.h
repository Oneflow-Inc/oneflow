#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_

#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/stream_scanner.h"

namespace oneflow {

class PersistentInStream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentInStream);

  PersistentInStream(fs::FileSystem* fs, const std::vector<std::string>& file_paths,
                     uint64_t offset, bool cyclic, bool with_local_copy);
  PersistentInStream(fs::FileSystem* fs, const std::vector<std::string>& file_paths, bool cyclic,
                     bool with_local_copy);
  PersistentInStream(fs::FileSystem* fs, const std::string& file_path, uint64_t offset, bool cyclic,
                     bool with_local_copy);
  PersistentInStream(fs::FileSystem* fs, const std::string& file_path, uint64_t offset);
  PersistentInStream(fs::FileSystem* fs, const std::string& file_path);

  // 0: success
  // -1: eof
  int32_t ReadLine(std::string* l);
  int32_t ReadFully(char* s, size_t n);

 private:
  bool IsEof() const;
  void UpdateBuffer();

  std::unique_ptr<StreamScanner> stream_scanner_;

  std::vector<char> buffer_;
  char* cur_buf_begin_;
  char* cur_buf_end_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENT_IN_STREAM_H_
