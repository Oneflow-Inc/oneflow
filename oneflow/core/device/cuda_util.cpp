#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

const char* CublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* CurandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

cudaDeviceProp global_device_prop;

}  // namespace

void InitGlobalCudaDeviceProp() { cudaGetDeviceProperties(&global_device_prop, 0); }

int32_t GetSMCudaMaxBlocksNum() {
  return global_device_prop.multiProcessorCount * global_device_prop.maxThreadsPerMultiProcessor
         / kCudaThreadsNumPerBlock;
}

template<>
void CudaCheck(cudaError_t error) {
  CHECK_EQ(error, cudaSuccess) << cudaGetErrorString(error);
}

template<>
void CudaCheck(cudnnStatus_t error) {
  CHECK_EQ(error, CUDNN_STATUS_SUCCESS) << cudnnGetErrorString(error);
}

template<>
void CudaCheck(cublasStatus_t error) {
  CHECK_EQ(error, CUBLAS_STATUS_SUCCESS) << CublasGetErrorString(error);
}

template<>
void CudaCheck(curandStatus_t error) {
  CHECK_EQ(error, CURAND_STATUS_SUCCESS) << CurandGetErrorString(error);
}

size_t GetAvailableGpuMemSize(int dev_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev_id);
  return prop.totalGlobalMem;
}

#ifdef PLATFORM_POSIX

namespace {

int32_t HexCharToInt(int32_t c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  } else if (c >= 'a' && c <= 'f') {
    return c - 'a';
  } else if (c >= 'A' && c <= 'F') {
    return c - 'A';
  } else {
    return -1;
  }
}

void MaskToCpuSet(const std::string& str, cpu_set_t* set) {
  CPU_ZERO_S(sizeof(cpu_set_t), set);
  int32_t cpu = 0;
  for (size_t pos = str.size() - 1; pos >= 0; --pos) {
    if (str.at(pos) == ',') { continue; }
    const int32_t val = HexCharToInt(str.at(pos));
    CHECK_NE(val, -1);
    if (val & 1) { CPU_SET_S(cpu, sizeof(cpu_set_t), set); }
    if (val & 2) { CPU_SET_S(cpu + 1, sizeof(cpu_set_t), set); }
    if (val & 4) { CPU_SET_S(cpu + 2, sizeof(cpu_set_t), set); }
    if (val & 8) { CPU_SET_S(cpu + 3, sizeof(cpu_set_t), set); }
    cpu += 4;
  }
}

}  // namespace

void NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size) {
  std::vector<char> pci_bus_id_buf(sizeof("0000:00:00.0") - 1);
  CudaCheck(
      cudaDeviceGetPCIBusId(pci_bus_id_buf.data(), static_cast<int>(pci_bus_id_buf.size()), dev));
  const std::string pci_bus_id(pci_bus_id_buf.data(), pci_bus_id_buf.size());
  const std::string pci_bus_id_short = pci_bus_id.substr(0, sizeof("0000:00") - 1);
  const std::string local_cpus_file =
      "/sys/class/pci_bus/" + pci_bus_id_short + "/../../" + pci_bus_id + "/local_cpus";
  char* path = realpath(local_cpus_file.c_str(), nullptr);
  CHECK_NOTNULL(path);
  std::ifstream is(path);
  std::string cpu_mask;
  CHECK(std::getline(is, cpu_mask).good());
  is.close();
  free(path);
  cpu_set_t new_cpu_set;
  MaskToCpuSet(cpu_mask, &new_cpu_set);
  cpu_set_t saved_cpu_set;
  sched_getaffinity(0, sizeof(cpu_set_t), &saved_cpu_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &new_cpu_set);
  CudaCheck(cudaMallocHost(ptr, size));
  sched_setaffinity(0, sizeof(cpu_set_t), &saved_cpu_set);
}

#endif

cudaDataType_t GetCudaDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cuda) \
  if (val == GetDataType<type_cpp>::value) { return type_cuda; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDA_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

#endif  // WITH_CUDA

}  // namespace oneflow
