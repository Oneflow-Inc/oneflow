#ifndef ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_
#define ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_

#include <string>
#include <memory>
#include <set>

namespace libkineto {

enum class ActivityType;
class ActivityTraceInterface;

}  // namespace libkineto

namespace oneflow {

namespace profiler {

enum class ActivityType {
  CPU = 0,
  CUDA,
};

using interface_trace_t = libkineto::ActivityTraceInterface;

struct ActivityTraceWrapper {
  explicit ActivityTraceWrapper(std::unique_ptr<interface_trace_t> trace);
  ActivityTraceWrapper() = default;
  ActivityTraceWrapper(ActivityTraceWrapper&&) = default;
  ActivityTraceWrapper(const ActivityTraceWrapper&) = delete;
  explicit operator bool() const;
  void save(const std::string& path);

  const std::unique_ptr<interface_trace_t>& get() { return trace_; }

 private:
  std::unique_ptr<interface_trace_t> trace_;
  bool saved_ = false;  // Kineto's save is destructive
};

using ActivitySet = std::set<ActivityType>;
void PrepareTrace(const bool cpuOnly, const ActivitySet& activities);
void StartTrace();
ActivityTraceWrapper StopTrace();

}  // namespace profiler
}  // namespace oneflow

#endif // ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_
