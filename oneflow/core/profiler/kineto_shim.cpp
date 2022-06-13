#include "oneflow/core/profiler/kineto_shim.h"
#include "libkineto.h"

namespace oneflow {

namespace profiler {
namespace {

const std::set<libkineto::ActivityType> cpuTypes{
    libkineto::ActivityType::CPU_OP,          libkineto::ActivityType::CPU_INSTANT_EVENT,
    libkineto::ActivityType::USER_ANNOTATION, libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::CUDA_RUNTIME,    libkineto::ActivityType::PYTHON_FUNCTION,
};

const std::set<libkineto::ActivityType> cudaTypes = {
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CONCURRENT_KERNEL,
    // CUDA_RUNTIME appears in both cpuTypes and cudaTypes.
    libkineto::ActivityType::CUDA_RUNTIME,
};
}  // namespace

const DeviceAndResource kineto_ids() {
  return {/*device=*/libkineto::processId(),
          /*resource=*/libkineto::systemThreadId()};
}

TraceWrapper::TraceWrapper(const int64_t start_time, const std::string& name)
    : cpu_trace_(std::make_unique<libkineto::CpuTraceBuffer>()) {
  cpu_trace_->span.startTime = start_time;
  cpu_trace_->gpuOpCount = -1;
  cpu_trace_->span.name = name;
}

void TraceWrapper::addCPUActivity(const std::string& name,
                                  const DeviceAndResource device_and_resource,
                                  const uint64_t correlation_id, const int64_t start_time,
                                  const int64_t end_time) {
  //   TORCH_CHECK((bool)(*this), "Cannot add event to non-existent trace.");
  cpu_trace_->activities.emplace_back(
      libkineto::GenericTraceActivity(cpu_trace_->span, libkineto::ActivityType::CPU_OP, name));
  auto& act = cpu_trace_->activities.back();
  act.device = device_and_resource.device;
  act.resource = device_and_resource.resource;
  act.id = correlation_id;
  act.startTime = start_time;
  act.endTime = end_time;
}

void TraceWrapper::transferCpuTrace(int64_t end_time) {
  cpu_trace_->span.endTime = end_time;
  libkineto::api().activityProfiler().transferCpuTrace(std::move(cpu_trace_));
}

TraceWrapper::operator bool() const { return cpu_trace_ != nullptr; }

ActivityTraceWrapper::ActivityTraceWrapper(std::unique_ptr<interface_trace_t> trace)
    : trace_(std::move(trace)), saved_{false} {}

ActivityTraceWrapper::operator bool() const { return trace_ != nullptr; }

void ActivityTraceWrapper::save(const std::string& path) {
  //   TORCH_CHECK(!saved_, "Trace is already saved.");
  //   TORCH_CHECK(trace_ != nullptr, "Missing trace.")
  trace_->save(path);
  saved_ = true;
}

void prepareTrace(const bool cpuOnly, const ActivitySet& activities) {
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

void startTrace() { libkineto::api().activityProfiler().startTrace(); }

ActivityTraceWrapper stopTrace() {
  return ActivityTraceWrapper{libkineto::api().activityProfiler().stopTrace()};
}

void pushCorrelationId(uint64_t correlation_id) {
  libkineto::api().activityProfiler().pushCorrelationId(correlation_id);
}

void popCorrelationId() { libkineto::api().activityProfiler().popCorrelationId(); }

void recordThreadInfo() { libkineto::api().activityProfiler().recordThreadInfo(); }

}  // namespace profiler
}  // namespace oneflow