#ifndef ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_
#define ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_

#include <bits/stdint-intn.h>
#include <string>
#include <memory>
#include <set>

namespace libkineto {

enum class ActivityType;
struct CpuTraceBuffer;
class ActivityTraceInterface;

}  // namespace libkineto

namespace oneflow {

namespace profiler {

enum class ActivityType {
  CPU = 0,
  CUDA,                   // CUDA kernels, runtime
  NUM_KINETO_ACTIVITIES,  // must be the last one
};

struct DeviceAndResource {
  int32_t device;
  int32_t resource;
};
const DeviceAndResource kineto_ids();

using trace_t = libkineto::CpuTraceBuffer;
using interface_trace_t = libkineto::ActivityTraceInterface;

struct TraceWrapper {
  TraceWrapper(const int64_t start_time, const std::string& name);
  TraceWrapper(TraceWrapper&&) = default;
  TraceWrapper(const TraceWrapper&) = delete;

  // The caller is expected to hold a mutex when calling `addCPUActivity` and
  // addMemoryUsageActivity.
  void addCPUActivity(const std::string& name, const DeviceAndResource device_and_resource,
                      const uint64_t correlation_id, const int64_t start_time,
                      const int64_t end_time);

  void transferCpuTrace(int64_t end_time);

  explicit operator bool() const;

  std::unique_ptr<trace_t>& get() { return cpu_trace_; }

 private:
  std::unique_ptr<trace_t> cpu_trace_;
};

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
void prepareTrace(const bool cpuOnly, const ActivitySet& activities);
void startTrace();
ActivityTraceWrapper stopTrace();
void pushCorrelationId(uint64_t correlation_id);
void popCorrelationId();
void recordThreadInfo();

}  // namespace profiler
}  // namespace oneflow

#endif // ONEFLOW_CORE_PROFILER_KINETO_SHIM_H_
