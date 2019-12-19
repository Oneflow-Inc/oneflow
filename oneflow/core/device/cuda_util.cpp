#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/platform.h"

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

void InitGlobalCudaDeviceProp() {
  cudaGetDeviceProperties(&global_device_prop, 0);
  if (IsCuda9OnTuringDevice()) {
    LOG(WARNING)
        << "CUDA 9 running on Turing device has known issues, consider upgrading to CUDA 10";
  }
}

int32_t GetSMCudaMaxBlocksNum() {
  int32_t n =
      global_device_prop.multiProcessorCount * global_device_prop.maxThreadsPerMultiProcessor;
  return (n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock;
}

bool IsCuda9OnTuringDevice() {
  return CUDA_VERSION >= 9000 && CUDA_VERSION < 9020 && global_device_prop.major == 7
         && global_device_prop.minor == 5;
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

void ParseCpuMask(const std::string& cpu_mask, cpu_set_t* cpu_set) {
  CPU_ZERO_S(sizeof(cpu_set_t), cpu_set);
  const char* const head = cpu_mask.c_str();
  const char* const tail = head + cpu_mask.size();
  const char* pos = head;
  std::vector<uint64_t> masks;
  while (pos < tail) {
    char* end_pos = nullptr;
    const uint64_t mask = std::strtoul(pos, &end_pos, 16);
    if (pos != head) {
      CHECK_EQ(end_pos - pos, 8);
    } else {
      CHECK_NE(end_pos, pos);
      CHECK_LE(end_pos - pos, 8);
    }
    if (end_pos < tail) { CHECK_EQ(*end_pos, ','); }
    masks.push_back(mask);
    pos = end_pos + 1;
  }
  int32_t cpu = 0;
  for (int64_t i = masks.size() - 1; i >= 0; i--) {
    for (uint64_t b = 0; b < 32; b++) {
      if ((masks.at(i) & (1UL << b)) != 0) { CPU_SET_S(cpu, sizeof(cpu_set_t), cpu_set); }
      cpu += 1;
    }
  }
}

std::string CudaDeviceGetCpuMask(int32_t dev_id) {
  std::vector<char> pci_bus_id_buf(sizeof("0000:00:00.0"));
  CudaCheck(cudaDeviceGetPCIBusId(pci_bus_id_buf.data(), static_cast<int>(pci_bus_id_buf.size()),
                                  dev_id));
  const std::string pci_bus_id(pci_bus_id_buf.data(), pci_bus_id_buf.size() - 1);
  const std::string pci_bus_id_short = pci_bus_id.substr(0, sizeof("0000:00") - 1);
  const std::string local_cpus_file =
      "/sys/class/pci_bus/" + pci_bus_id_short + "/device/" + pci_bus_id + "/local_cpus";
  char* cpu_map_path = realpath(local_cpus_file.c_str(), nullptr);
  CHECK_NOTNULL(cpu_map_path);
  std::ifstream is(cpu_map_path);
  std::string cpu_mask;
  CHECK(std::getline(is, cpu_mask).good());
  is.close();
  free(cpu_map_path);
  return cpu_mask;
}

void CudaDeviceGetCpuAffinity(int32_t dev_id, cpu_set_t* cpu_set) {
  const std::string cpu_mask = CudaDeviceGetCpuMask(dev_id);
  ParseCpuMask(cpu_mask, cpu_set);
}

}  // namespace

#endif

void NumaAwareCudaMallocHost(int32_t dev, void** ptr, size_t size) {
#ifdef PLATFORM_POSIX
  cpu_set_t new_cpu_set;
  CudaDeviceGetCpuAffinity(dev, &new_cpu_set);
  cpu_set_t saved_cpu_set;
  CHECK_EQ(sched_getaffinity(0, sizeof(cpu_set_t), &saved_cpu_set), 0);
  CHECK_EQ(sched_setaffinity(0, sizeof(cpu_set_t), &new_cpu_set), 0);
  CudaCheck(cudaMallocHost(ptr, size));
  CHECK_EQ(sched_setaffinity(0, sizeof(cpu_set_t), &saved_cpu_set), 0);
#else
  UNIMPLEMENTED();
#endif
}

cudaDataType_t GetCudaDataType(DataType val) {
#define MAKE_ENTRY(type_cpp, type_cuda) \
  if (val == GetDataType<type_cpp>::value) { return type_cuda; }
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, CUDA_DATA_TYPE_SEQ);
#undef MAKE_ENTRY
  UNIMPLEMENTED();
}

CudaCurrentDeviceGuard::CudaCurrentDeviceGuard(int32_t dev_id) {
  CudaCheck(cudaGetDevice(&saved_dev_id_));
  CudaCheck(cudaSetDevice(dev_id));
}

CudaCurrentDeviceGuard::~CudaCurrentDeviceGuard() { CudaCheck(cudaSetDevice(saved_dev_id_)); }

#endif  // WITH_CUDA

}  // namespace oneflow
