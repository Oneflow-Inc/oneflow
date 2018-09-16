#ifndef ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_
#define ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class LogStreamDestination final {
 public:
  LogStreamDestination(fs::FileSystem* file_system, const std::string& base_dir)
      : file_system_(file_system), base_dir_(base_dir) {}
  ~LogStreamDestination() = default;
  fs::FileSystem* mut_file_system() const { return file_system_; };
  const std::string& base_dir() const { return base_dir_; };

 private:
  fs::FileSystem* file_system_;
  std::string base_dir_;
};

class TeePersistentLogStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TeePersistentLogStream);
  explicit TeePersistentLogStream(const std::string& path);
  ~TeePersistentLogStream();

  void Write(const char* s, size_t n);

 private:
  void Flush();
  std::vector<LogStreamDestination> destinations_;
  std::vector<std::unique_ptr<PersistentOutStream>> branches_;
};

template<typename T>
typename std::enable_if<std::is_fundamental<T>::value, TeePersistentLogStream&>::type operator<<(
    TeePersistentLogStream& log_stream, const T& x) {
  const char* x_ptr = reinterpret_cast<const char*>(&x);
  size_t n = sizeof(x);
  log_stream.Write(x_ptr, n);
  return log_stream;
}

inline TeePersistentLogStream& operator<<(TeePersistentLogStream& log_stream,
                                          const std::string& s) {
  log_stream.Write(s.c_str(), s.size());
  return log_stream;
}

template<size_t n>
TeePersistentLogStream& operator<<(TeePersistentLogStream& log_stream, const char (&s)[n]) {
  log_stream.Write(s, strlen(s));
  return log_stream;
}

void SaveProtoAsTextFile(const PbMessage& proto, const std::string& path);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_TEE_PERSISTENT_LOG_STREAM_H_
