#ifndef _MEMORY_USAGE_H_
#define _MEMORY_USAGE_H_
namespace caffe {
// NOTE(jiyuan): all the memory in PS logic should belong to the model path;
struct MemoryUsage {
  size_t device_memory_size_of_data_path{ 0 };
  size_t host_memory_size_of_data_path{ 0 };
  size_t device_memory_size_of_model_path{ 0 };
  size_t host_memory_size_of_model_path{ 0 };
  void Add(const MemoryUsage& other) {
    device_memory_size_of_data_path += other.device_memory_size_of_data_path;
    host_memory_size_of_data_path += other.host_memory_size_of_data_path;
    device_memory_size_of_model_path += other.device_memory_size_of_model_path;
    host_memory_size_of_model_path += other.host_memory_size_of_model_path;
  }
};
}  // namespace caffe
#endif  // _MEMORY_USAGE_H_
