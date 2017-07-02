#include "oneflow/core/kernel/clone_kernel.h"
#include <random>
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/device/cuda_device_context.h"

namespace oneflow {

namespace {

enum class Location { kHost, kDevice };

struct CloneKernelBlobs {
  CloneKernelBlobs(const int);
  int out_num;
  Blob* in_blob;
  Blob* in_diff_blob;
  std::vector<Blob*> out_blobs;
  std::vector<Blob*> out_diff_blobs;
};

CloneKernelBlobs::CloneKernelBlobs(const int out_num_param)
    : out_num(out_num_param),
      in_blob(nullptr),
      in_diff_blob(nullptr),
      out_blobs(out_num_param, nullptr),
      out_diff_blobs(out_num_param, nullptr) {}

template<typename FloatingPointType>
Blob* CreateBlob(const std::vector<int64_t>& dim_vec,
                 const FloatingPointType* data_vec, Location location) {
  void* dptr = nullptr;
  Shape* shape = new Shape(dim_vec);

  size_t dptr_size = shape->elem_cnt() * sizeof(FloatingPointType);
  if (location == Location::kHost) {
    CHECK_EQ(cudaMallocHost(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToHost),
             cudaSuccess);
  } else {
    CHECK_EQ(cudaMalloc(&dptr, dptr_size), cudaSuccess);
    CHECK_EQ(cudaMemcpy(dptr, data_vec, dptr_size, cudaMemcpyHostToDevice),
             cudaSuccess);
  }
  return new Blob(dptr, shape);
}

template<typename FloatingPointType>
Blob* CreateBlobWithSameValue(const std::vector<int64_t>& dim_vec,
                              FloatingPointType value, Location location) {
  Shape* shape = new Shape(dim_vec);
  FloatingPointType* data_vec = new FloatingPointType[shape->elem_cnt()];
  std::fill(data_vec, data_vec + shape->elem_cnt(), value);
  return CreateBlob<FloatingPointType>(dim_vec, data_vec, location);
}

template<typename FloatingPointType>
Blob* CreateBlobWithRandomValue(const std::vector<int64_t>& dim_vec,
                                Location location) {
  Shape* shape = new Shape(dim_vec);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<FloatingPointType> dis(0, 10);
  FloatingPointType* data_vec = new FloatingPointType[shape->elem_cnt()];
  for (int64_t i = 0; i != shape->elem_cnt(); ++i) { data_vec[i] = dis(gen); }
  return CreateBlob<FloatingPointType>(dim_vec, data_vec, location);
}

template<DeviceType device_type, typename FloatingPointType>
Kernel* ConstructCloneKernel(const int out_num, const std::string& lbn) {
  OperatorConf op_conf;
  op_conf.set_name("clone_test");
  CloneOpConf* clone_conf = op_conf.mutable_clone_conf();
  clone_conf->set_out_num(out_num);
  clone_conf->set_lbn(lbn);
  auto clone_op = OpMgr::Singleton().ConstructOp(op_conf);

  OperatorProto op_proto;
  clone_op->ToProto(&op_proto);

  auto clone_kernel = new CloneKernel<device_type, FloatingPointType>();
  clone_kernel->InitFromOpProto(op_proto);

  return clone_kernel;
}

template<typename FloatingPointType>
CloneKernelBlobs* CreateCloneKernelBlobsFromData(
    const std::vector<int64_t>& dim_vec, const int out_num,
    const HashMap<std::string, FloatingPointType>& init_value_map,
    Location location) {
  CloneKernelBlobs* ck_blobs = new CloneKernelBlobs(out_num);
  ck_blobs->in_blob = CreateBlobWithSameValue<FloatingPointType>(
      dim_vec, init_value_map.at("in_blob"), location);
  ck_blobs->in_diff_blob = CreateBlobWithSameValue<FloatingPointType>(
      dim_vec, init_value_map.at("in_diff_blob"), location);
  // 0 : {k. k. k...}
  // 1 : {2k, 2k, 2k...}
  // ....
  for (int i = 0; i != out_num; ++i) {
    ck_blobs->out_blobs[i] = CreateBlobWithSameValue<FloatingPointType>(
        dim_vec, init_value_map.at("out_blob") * (i + 1), location);
    ck_blobs->out_diff_blobs[i] = CreateBlobWithSameValue<FloatingPointType>(
        dim_vec, init_value_map.at("out_diff_blob") * (i + 1), location);
  }
  return ck_blobs;
}

template<typename FloatingPointType>
CloneKernelBlobs* CreateCloneKernelBlobsWithRandomValue(
    const std::vector<int64_t>& dim_vec, const int out_num, Location location) {
  CloneKernelBlobs* ck_blobs = new CloneKernelBlobs(out_num);
  ck_blobs->in_blob =
      CreateBlobWithRandomValue<FloatingPointType>(dim_vec, location);
  ck_blobs->in_diff_blob =
      CreateBlobWithRandomValue<FloatingPointType>(dim_vec, location);
  for (int i = 0; i != out_num; ++i) {
    ck_blobs->out_blobs[i] =
        CreateBlobWithSameValue<FloatingPointType>(dim_vec, 0.0f, location);
    ck_blobs->out_diff_blobs[i] =
        CreateBlobWithRandomValue<FloatingPointType>(dim_vec, location);
  }
  return ck_blobs;
}

void InitBn2BlobPtr(HashMap<std::string, Blob*>& bn2blob_ptr,
                    CloneKernelBlobs* ck_blobs) {
  bn2blob_ptr["in"] = ck_blobs->in_blob;
  bn2blob_ptr["in_diff"] = ck_blobs->in_diff_blob;
  for (size_t i = 0; i != ck_blobs->out_num; ++i) {
    bn2blob_ptr["out_" + std::to_string(i)] = ck_blobs->out_blobs[i];
    bn2blob_ptr["out_" + std::to_string(i) + "_diff"] =
        ck_blobs->out_diff_blobs[i];
  }
}

template<typename FloatingPointType>
void CPUStreamExec(int out_num, std::function<Blob*(const std::string&)> fp) {
  KernelCtx ctx;
  ctx.device_ctx = new CpuDeviceCtx(new CpuStream);
  auto clone_kernel = ConstructCloneKernel<DeviceType::kCPU, FloatingPointType>(
      out_num, "clone_kernel_test");

  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);

  auto cpu_thread = std::thread([&] {
    std::function<void()> work;
    // Both Forward and Backward receive out_num times
    for (int i = 0; i < out_num * 2; ++i) {
      if (ctx.device_ctx->cpu_stream()->ReceiveWork(&work) == 0) { work(); }
    }
  });
  cpu_thread.join();
}

template<typename FloatingPointType>
void GPUStreamExec(int out_num, std::function<Blob*(const std::string&)> fp) {
  cudaStream_t cuda_stream;
  CHECK_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  cublasHandle_t cublas_handle;
  CHECK_EQ(cublasCreate(&cublas_handle), CUBLAS_STATUS_SUCCESS);
  KernelCtx ctx;
  ctx.device_ctx = new CudaDeviceCtx(&cuda_stream, &cublas_handle, nullptr);
  auto clone_kernel = ConstructCloneKernel<DeviceType::kGPU, FloatingPointType>(
      out_num, "clone_kernel_test");

  clone_kernel->Forward(ctx, fp);
  clone_kernel->Backward(ctx, fp);

  CHECK_EQ(cudaStreamSynchronize(ctx.device_ctx->cuda_stream()), cudaSuccess);
}

void* GetMemcpyDeviceToHost(const void* src, size_t count) {
  void* dst = malloc(count);
  CHECK_EQ(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost), cudaSuccess);
  return dst;
}

template<typename FloatingPointType>
void TestForward(CloneKernelBlobs* ck_blobs, Location location) {
  const int64_t dptr_size =
      ck_blobs->in_blob->shape().elem_cnt() * sizeof(FloatingPointType);
  void* in_blob_dptr = nullptr;
  std::vector<void*> out_blobs_dptr(ck_blobs->out_num);
  if (location == Location::kHost) {
    in_blob_dptr = ck_blobs->in_blob->mut_dptr();
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_blobs_dptr[i] = ck_blobs->out_blobs[i]->mut_dptr();
    }
  } else {
    in_blob_dptr = GetMemcpyDeviceToHost(ck_blobs->in_blob->dptr(), dptr_size);
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_blobs_dptr[i] =
          GetMemcpyDeviceToHost(ck_blobs->out_blobs[i]->dptr(), dptr_size);
    }
  }
  for (int i = 0; i != ck_blobs->out_num; ++i) {
    ASSERT_EQ(memcmp(in_blob_dptr, out_blobs_dptr[i], dptr_size), 0);
  }
}

template<typename FloatingPointType>
void TestBackward(CloneKernelBlobs* ck_blobs, Location location) {
  const int64_t dptr_size =
      ck_blobs->in_blob->shape().elem_cnt() * sizeof(FloatingPointType);
  FloatingPointType* in_diff_blob_dptr = nullptr;
  std::vector<FloatingPointType*> out_diff_blob_dptrs(ck_blobs->out_num,
                                                      nullptr);
  if (location == Location::kHost) {
    in_diff_blob_dptr =
        static_cast<FloatingPointType*>(ck_blobs->in_diff_blob->mut_dptr());
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_diff_blob_dptrs[i] = static_cast<FloatingPointType*>(
          ck_blobs->out_diff_blobs[i]->mut_dptr());
    }
  } else {
    in_diff_blob_dptr = static_cast<FloatingPointType*>(
        GetMemcpyDeviceToHost(ck_blobs->in_diff_blob->dptr(), dptr_size));
    for (int i = 0; i != ck_blobs->out_num; ++i) {
      out_diff_blob_dptrs[i] =
          static_cast<FloatingPointType*>(GetMemcpyDeviceToHost(
              ck_blobs->out_diff_blobs[i]->dptr(), dptr_size));
    }
  }
  for (int64_t i = 0; i != ck_blobs->in_blob->shape().elem_cnt(); ++i) {
    FloatingPointType sum = {0.0f};
    for (int out_j = 0; out_j != ck_blobs->out_num; ++out_j) {
      sum += out_diff_blob_dptrs[out_j][i];
    }
    ASSERT_FLOAT_EQ(sum, in_diff_blob_dptr[i]);
  }
}

template<typename FloatingPointType>
void CloneKernelTest(CloneKernelBlobs* ck_blobs, Location location) {
  HashMap<std::string, Blob*> bn2blob_ptr;
  InitBn2BlobPtr(bn2blob_ptr, ck_blobs);
  auto fp = [&bn2blob_ptr](const std::string& bn) {
    return bn2blob_ptr.at(bn);
  };
  if (location == Location::kHost) {
    CPUStreamExec<FloatingPointType>(ck_blobs->out_num, fp);
  } else {
    GPUStreamExec<FloatingPointType>(ck_blobs->out_num, fp);
  }
  TestForward<FloatingPointType>(ck_blobs, location);
  TestBackward<FloatingPointType>(ck_blobs, location);
}

}  // namespace

template<typename FloatingPointType>
void TestRandomData(const std::vector<int64_t> dim_vec, int out_num,
                    Location location) {
  CloneKernelBlobs* ck_blobs =
      CreateCloneKernelBlobsWithRandomValue<FloatingPointType>(dim_vec, out_num,
                                                               location);
  CloneKernelTest<FloatingPointType>(ck_blobs, location);
}

template<typename FloatingPointType>
void TestFixedData(const std::vector<int64_t> dim_vec, int out_num,
                   Location location) {
  const HashMap<std::string, FloatingPointType> init_value_map = {
      {"in_blob", 1.3},
      {"out_blob", 2.4},
      {"in_diff_blob", 3.5},
      {"out_diff_blob", 4.6}};
  CloneKernelBlobs* ck_blobs =
      CreateCloneKernelBlobsFromData<FloatingPointType>(
          dim_vec, out_num, init_value_map, location);
  CloneKernelTest<FloatingPointType>(ck_blobs, location);
}

TEST(CloneKernel, random_4x5x6x7) {
  const int out_num = 3;
  const std::vector<int64_t> dim_vec = {4, 5, 6, 7};
  TestRandomData<float>(dim_vec, out_num, Location::kHost);
  TestRandomData<double>(dim_vec, out_num, Location::kHost);
  TestRandomData<float>(dim_vec, out_num, Location::kDevice);
  TestRandomData<double>(dim_vec, out_num, Location::kDevice);
}

TEST(CloneKernel, fixed_1x3x2) {
  const int out_num = 3;
  const std::vector<int64_t> dim_vec = {1, 3, 2};
  TestFixedData<float>(dim_vec, out_num, Location::kHost);
  TestFixedData<double>(dim_vec, out_num, Location::kHost);
  TestFixedData<float>(dim_vec, out_num, Location::kDevice);
  TestFixedData<double>(dim_vec, out_num, Location::kDevice);
}

}  // namespace oneflow
