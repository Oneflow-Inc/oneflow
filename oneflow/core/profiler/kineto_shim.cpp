#include "oneflow/core/profiler/kineto_shim.h"
#include "libkineto.h"

namespace oneflow {

namespace profiler {
namespace {

const std::set<libkineto::ActivityType> cpuTypes{
    libkineto::ActivityType::CPU_OP,          libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION, libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,  // something like cudaLaunchKernel
    libkineto::ActivityType::PYTHON_FUNCTION,
};

const std::set<libkineto::ActivityType> cudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY, libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,  // cuda kernel
    // CUDA_RUNTIME appears in both cpuTypes and cudaTypes.
    libkineto::ActivityType::CUDA_RUNTIME,  // something like cudaLaunchKernel
};
}  // namespace

ActivityTraceWrapper::ActivityTraceWrapper(std::unique_ptr<interface_trace_t> trace)
    : trace_(std::move(trace)), saved_{false} {}

ActivityTraceWrapper::operator bool() const { return trace_ != nullptr; }

void ActivityTraceWrapper::save(const std::string& path) {
  //   TORCH_CHECK(!saved_, "Trace is already saved.");
  //   TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
}

void PrepareTrace(const bool cpuOnly, const ActivitySet& activities) {
  if (!libkineto::api().isProfilerRegistered()) {
    libkineto_init(/*cpuOnly=*/cpuOnly, /*logOnError=*/true);
    libkineto::api().suppressLogMessages();
  }

  if (!libkineto::api().isProfilerInitialized()) { libkineto::api().initProfilerIfRegistered(); }

  std::set<libkineto::ActivityType> k_activities;
  if (activities.count(ActivityType::CPU)) {
    k_activities.insert(cpuTypes.begin(), cpuTypes.end());
  }
  if (activities.count(ActivityType::CUDA)) {
    k_activities.insert(cudaTypes.begin(), cudaTypes.end());
  }

  libkineto::api().activityProfiler().prepareTrace(k_activities);
}

void StartTrace() { libkineto::api().activityProfiler().startTrace(); }

ActivityTraceWrapper StopTrace() {
  return ActivityTraceWrapper{libkineto::api().activityProfiler().stopTrace()};
}

}  // namespace profiler
}  // namespace oneflow