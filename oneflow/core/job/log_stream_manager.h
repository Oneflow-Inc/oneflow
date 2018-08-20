#ifndef ONEFLOW_CORE_JOB_LOG_STREAM_MANAGER_H_
#define ONEFLOW_CORE_JOB_LOG_STREAM_MANAGER_H_

#include <utility>
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/out_stream.h"

namespace oneflow {

class LogStreamMgr final {
 public:
  std::unique_ptr<OutStream> Create(const std::string &path);

 private:
  class LogStreamMgrSink final {
   public:
    LogStreamMgrSink(fs::FileSystem *file_system, std::string prefix)
        : file_system_(file_system),
          prefix_(std::move(prefix)){

          };

    LogStreamMgrSink() = delete;

    fs::FileSystem *file_system() { return file_system_; };

    std::string &prefix() { return prefix_; };

   private:
    fs::FileSystem *file_system_;
    std::string prefix_;
  };

 private:
  friend class Global<LogStreamMgr>;
  LogStreamMgr();

  ~LogStreamMgr() = default;

  std::vector<LogStreamMgrSink> sinks;

  std::string GenerateLogDirNameFromContext();
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LOG_STREAM_MANAGER_H_
