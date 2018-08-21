#ifndef ONEFLOW_CORE_JOB_LOG_STREAM_MANAGER_H_
#define ONEFLOW_CORE_JOB_LOG_STREAM_MANAGER_H_

#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/log_stream.h"

namespace oneflow {

class LogStreamMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogStreamMgr);
  std::unique_ptr<LogStream> Create(const std::string& path) const;

 private:
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

 private:
  friend class Global<LogStreamMgr>;
  LogStreamMgr();
  ~LogStreamMgr() = default;
  const std::string GenerateLogDirNameFromContext() const;

  std::vector<LogStreamDestination> destinations_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_LOG_STREAM_MANAGER_H_
