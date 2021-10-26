#include "oneflow/core/device/stream_index.h"

namespace oneflow {

StreamIndexGenerator::StreamIndexGenerator() : next_stream_index_(0) {}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::GenerateStreamIndex() {
  return next_stream_index_.fetch_add(1, std::memory_order_relaxed);
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::GenerateStreamIndex(
    const std::string& name, size_t num) {
  CHECK_GT(num, 0);
  std::unique_lock<std::mutex> lck1(named_rr_range_mutex_);
  auto range_it = name2round_robin_range_.find(name);
  if (range_it == name2round_robin_range_.end()) {
    stream_index_t cur_stream_index = next_stream_index_.fetch_add(1, std::memory_order_relaxed);
    range_it = name2round_robin_range_.emplace(name, std::make_pair(cur_stream_index, num)).first;
  } else {
    CHECK_EQ(range_it->second.second, num);
  }

  stream_index_t cur_stream_index = range_it->second.first;
  if (num > 1) {
    std::unique_lock<std::mutex> lck2(named_rr_offset_mutex_);
    auto offset_it = name2round_robine_offset.find(name);
    if (offset_it == name2round_robine_offset.end()) {
      offset_it = name2round_robine_offset.emplace(name, 0).first;
    }
    cur_stream_index += offset_it->second++;
    if (offset_it->second > range_it->second.second) { offset_it->second = 0; }
  }
  return cur_stream_index;
}

StreamIndexGenerator* StreamIndexGeneratorManager::GetOrCreateGenerator(const DeviceId& device_id) {
  std::unique_lock<std::mutex> lck(mtx_);
  auto iter = generators_.find(device_id);
  if (iter == generators_.end()) {
    std::unique_ptr<StreamIndexGenerator> generaotr_ptr(new StreamIndexGenerator());
    iter = generators_.emplace(device_id, std::move(generaotr_ptr)).first;
  }
  return iter->second.get();
}

}  // namespace oneflow
