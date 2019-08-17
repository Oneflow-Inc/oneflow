#include <cuda_runtime.h>

namespace oneflow {
  
struct CudaLaunchConfig {
  // Logical number of thread that works one the elements. If each logical thread
  // works on exactly a single element, this is the same as the working element count.
  int32_t virtual_thread_count = -1;

  int32_t threads_per_block = -1;
  int32_t block_count = -1;
};

struct CudaDeviceProperty {
  // # of GPU's multi processors
  int32_t multi_processors_num = -1;
  int32_t max_threads_per_multiprocessor = -1;
  int32_t max_threads_per_block = -1;
};

inline CudaDeviceProperty GetCudaDeviceProp(){
  int32_t device_id = 0;
  cudaSetDevice(device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);

  CudaDeviceProperty prop;
  prop.multi_processors_num = device_prop.multiProcessorCount;
  prop.max_threads_per_multiprocessor = device_prop.maxThreadsPerMultiProcessor;
  prop.max_threads_per_block = device_prop.maxThreadsPerBlock;
  return prop;
}

inline CudaLaunchConfig GetCudaLaunchConfig(int32_t work_elem_count){
  CHECK_GT(work_elem_count, 0);
  auto cuda_device_prop = GetCudaDeviceProp();
  const int32_t virtual_thread_count = work_elem_count;
  const int32_t physical_thread_count = std::min(
       cuda_device_prop.multi_processors_num * cuda_device_prop.max_threads_per_multiprocessor,
       virtual_thread_count);
  const int32_t threads_per_block = std::min(1024, prop.max_threads_per_block);
  auto div_up = [] (int a, int b) { return (a + b -1) / b; }
  const int32_t block_count = std::min(div_up(physical_thread_count, threads_per_block),
                                       prop.multi_processors_num);
  auto cuda_launch_config = CudaLaunchConfig();
  cuda_launch_config.virtual_thread_count = virtual_thread_count;
  cuda_launch_config.threads_per_block = therads_per_block;
  cuda_launch_config.block_count = block_count;
  return cuda_launch_config;

}

} // namespace oneflow 
