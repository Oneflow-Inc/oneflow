#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "cuda_runtime.h"

#include "oneflow/xrt/tensorrt/trt_int8_calibrator.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

void TRTInt8Calibrator::setBatchSize(const int batch_size) {  // NOLINT
  batch_size_ = batch_size;
}

// set the batch size before constructing the thread to execute engine
int TRTInt8Calibrator::getBatchSize() const { return batch_size_; }

TRTInt8Calibrator::TRTInt8Calibrator()  // NOLINT
    : done_(false), calib_running_(true), batch_is_set_(false) {}

TRTInt8Calibrator::TRTInt8Calibrator(const std::string& calib_data)
    : batch_size_(0),
      done_(true),
      calib_running_(false),
      batch_is_set_(false),
      calibration_table_(calib_data) {}

void TRTInt8Calibrator::waitAndSetDone() {
  std::unique_lock<std::mutex> lk(cond_mtx_);
  while ((calib_running_ || batch_is_set_) && !done_) cond_.wait(lk);
  if (!done_) {
    done_ = true;
    cond_.notify_all();
  }
}

// There might be more than one input for trt subgraph,
// So, we use a map to store input information.
bool TRTInt8Calibrator::setBatch(const std::vector<const Parameter*>& params) {
  std::unique_lock<std::mutex> lk(cond_mtx_);
  //  There is a producer and a consumer. The producer set the batch data and
  //  the consumer get the batch data. The size of the data pool is one.
  //  So, the producer has to wait for the consumer to finish processing before
  //  they can set the data.
  while ((calib_running_ || batch_is_set_) && (!done_)) cond_.wait(lk);
  // The done_ is set to true using waitAndSetDone, When all calibration data
  // are processed.
  if (done_) return false;

  // Sets the batch.
  for (const auto& it : params) {
    auto dataptr = dev_buffers_.find(it->name());
    if (dataptr == dev_buffers_.end()) {
      LOG(FATAL) << "Input name '" << it->name()  // NOLINT
                 << "' does not match with the buffer names";
    }
    const auto& d = dataptr->second;
    CHECK_EQ(cudaSuccess,                               // NOLINT
             cudaMemcpy(d.first, it->data(), d.second,  // NOLINT
                        cudaMemcpyDeviceToDevice))      // NOLINT
        << "Fail to cudaMemcpy for " << it->name();
  }

  batch_is_set_ = true;
  cond_.notify_all();
  return true;
}

bool TRTInt8Calibrator::getBatch(void** bindings, const char** names,  // NOLINT
                                 int num_bindings) {
  std::unique_lock<std::mutex> lk(cond_mtx_);
  // Notify finish of last round of calibration.
  calib_running_ = false;
  cond_.notify_all();

  // As long as there is data in the pool, the consumer can get it.
  while (!batch_is_set_ && !done_) cond_.wait(lk);
  if (done_) return false;

  // Gets the batch
  for (int i = 0; i < num_bindings; i++) {
    auto it = dev_buffers_.find(names[i]);
    if (it == dev_buffers_.end()) {
      LOG(FATAL) << "Calibration engine asked for unknown tensor name '"  // NOLINT
                 << names[i] << "' at position " << i;
    }
    bindings[i] = it->second.first;
  }

  batch_is_set_ = false;
  calib_running_ = true;
  return true;
}

void TRTInt8Calibrator::setDone() {
  std::unique_lock<std::mutex> lk(cond_mtx_);
  done_ = true;
  cond_.notify_all();
}

const void* TRTInt8Calibrator::readCalibrationCache(size_t& length) {
  if (calibration_table_.empty()) return nullptr;
  length = calibration_table_.size();
  return calibration_table_.data();
}

void TRTInt8Calibrator::writeCalibrationCache(const void* ptr,  // NOLINT
                                              std::size_t length) {
  calibration_table_ = std::string((const char*)ptr, length);
}

/*static*/ TRTInt8CalibratorResource*  // NOLINT
TRTInt8CalibratorResource::LookupOrCreate(const std::string& name) {
  static std::mutex mutex;
  static std::unordered_map<std::string,  // NOLINT
                            TRTInt8CalibratorResource*>
      resources;
  auto it = resources.find(name);
  if (it == resources.end()) {
    std::lock_guard<std::mutex> lock(mutex);
    while (resources.find(name) == resources.end()) {
      it = resources.emplace(name, new TRTInt8CalibratorResource).first;
    }
  }
  return it->second;
}

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow
