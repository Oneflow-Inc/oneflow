#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/device/cpu_device_context.h"

namespace oneflow {

namespace {

template<typename T>
void InitializeWithConf(const InitializerConf& conf, const uint32_t random_seed, Blob* blob) {
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, conf, random_seed, blob);
}

struct InitializeWithConfUtil final {
#define MAKE_INITIALIZE_SWITCH_ENTRY(func_name, T) func_name<T>
  DEFINE_STATIC_SWITCH_FUNC(void, InitializeWithConf, MAKE_INITIALIZE_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ));
#undef MAKE_INITIALIZE_SWITCH_ENTRY
};

TensorSliceView GetPartSlice(const KernelConf& kernel_conf, const int64_t parallel_id) {
  const ModelIoV2KernelConf& conf = kernel_conf.model_io_v2_conf();
  return TensorSliceView(conf.slice_view(parallel_id));
}

TensorSliceView GetPartSlice(const KernelConf& kernel_conf) {
  return GetPartSlice(kernel_conf, kernel_conf.model_io_v2_conf().parallel_ctx().parallel_id());
}

class OnDemandHostBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OnDemandHostBlob);
  explicit OnDemandHostBlob(const Blob* like) {
    blob_desc_.reset(new RtBlobDesc(like->blob_desc().blob_desc_proto()));
    Init();
  }
  explicit OnDemandHostBlob(const RtBlobDesc& blob_desc) {
    blob_desc_.reset(new RtBlobDesc(blob_desc.blob_desc_proto()));
    Init();
  }
  explicit OnDemandHostBlob(const Shape& shape, DataType data_type) {
    BlobDesc blob_desc(data_type);
    blob_desc.mut_shape() = shape;
    blob_desc_.reset(new RtBlobDesc(blob_desc));
    Init();
  }
  ~OnDemandHostBlob() = default;

  Blob* blob() const { return blob_.get(); }

 private:
  void Init() {
    data.resize(blob_desc_->TotalByteSize());
    blob_.reset(new Blob(nullptr, blob_desc_.get(), data.data()));
  }

  std::vector<char> data;
  std::unique_ptr<Blob> blob_;
  std::unique_ptr<RtBlobDesc> blob_desc_;
};

template<DeviceType device_type>
void SyncCopyToHost(DeviceCtx* ctx, const void* src, void* dst, size_t size);

template<>
void SyncCopyToHost<DeviceType::kCPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
}

#ifdef WITH_CUDA
template<>
void SyncCopyToHost<DeviceType::kGPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
  CudaCheck(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, ctx->cuda_stream()));
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
}
#endif

template<DeviceType device_type>
void SyncCopyToDevice(DeviceCtx* ctx, const void* src, void* dst, size_t size);

template<>
void SyncCopyToDevice<DeviceType::kCPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  std::memcpy(dst, src, size);
}

#ifdef WITH_CUDA
template<>
void SyncCopyToDevice<DeviceType::kGPU>(DeviceCtx* ctx, const void* src, void* dst, size_t size) {
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
  CudaCheck(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, ctx->cuda_stream()));
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));
}
#endif

template<DeviceType device_type>
std::string SyncReadStringFromBlob(DeviceCtx* ctx, const Blob* blob) {
  std::vector<char> content;
  const int64_t size = blob->shape().elem_cnt();
  content.resize(size);
  SyncCopyToHost<device_type>(ctx, blob->dptr(), content.data(), size);
  return std::string(content.data(), content.size());
}

std::string GetTmpPartKey(const std::string& base, const int64_t parallel_id,
                          const int64_t parallel_num) {
  return "tmp-part-" + std::to_string(parallel_id) + "-" + std::to_string(parallel_num) + "-"
         + base;
}

std::string GetTmpPartKey(const std::string& base, const ParallelContext& parallel_ctx) {
  return GetTmpPartKey(base, parallel_ctx.parallel_id(), parallel_ctx.parallel_num());
}

void HostSliceCopy(Blob* dst, const TensorSliceView& dst_slice, const Blob* src,
                   const TensorSliceView& src_slice) {
  CpuDeviceCtx cpu_device_ctx;
  std::unique_ptr<MemoryCopier> host_memory_copier(NewDefaultMemoryCopier(DeviceType::kCPU));
  TensorSliceCopier copier(dst_slice, src_slice, dst->data_type());
  copier.Copy(&cpu_device_ctx, *host_memory_copier, dst, src);
}

template<DeviceType device_type>
class AutoSyncBlobAccessor final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoSyncBlobAccessor);
  AutoSyncBlobAccessor(DeviceCtx* ctx, Blob* underlying, bool read_sync, bool write_sync)
      : device_ctx_(ctx),
        underlying_(underlying),
        read_sync_(read_sync),
        write_sync_(write_sync),
        host_blob_(underlying) {
    if (read_sync_) {
      SyncCopyToHost<device_type>(device_ctx_, underlying_->dptr(), host_blob_.blob()->mut_dptr(),
                                  underlying_->ByteSizeOfDataContentField());
    }
  }
  ~AutoSyncBlobAccessor() {
    if (write_sync_) {
      SyncCopyToDevice<device_type>(device_ctx_, host_blob_.blob()->dptr(), underlying_->mut_dptr(),
                                    underlying_->ByteSizeOfDataContentField());
    }
  }

  Blob* host_blob() { return host_blob_.blob(); }

 private:
  DeviceCtx* device_ctx_;
  Blob* underlying_;
  bool read_sync_;
  bool write_sync_;
  OnDemandHostBlob host_blob_;
};

template<>
class AutoSyncBlobAccessor<DeviceType::kCPU> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoSyncBlobAccessor);
  AutoSyncBlobAccessor(DeviceCtx* ctx, Blob* underlying, bool read_sync, bool write_sync)
      : underlying_(underlying) {}
  ~AutoSyncBlobAccessor() = default;

  Blob* host_blob() { return underlying_; }

 private:
  Blob* underlying_;
};

}  // namespace

template<DeviceType device_type>
class ModelInitV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelInitV2Kernel);
  ModelInitV2Kernel() = default;
  ~ModelInitV2Kernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelInitV2OpConf& conf = this->op_conf().model_init_v2_conf();
    Blob* ref = BnInOp2Blob("ref");
    const VariableOpConf& original_variable_conf = conf.original_variable_conf();
    const Shape logical_blob_shape(original_variable_conf.shape());
    const DataType data_type = ref->data_type();
    const std::string& var_lbn =
        GenLogicalBlobName(conf.variable_op_name(), original_variable_conf.out());
    const TensorSliceView slice = GetPartSlice(this->kernel_conf());
    AutoSyncBlobAccessor<device_type> ref_accessor(ctx.device_ctx, ref, false, true);
    if (original_variable_conf.has_initializer()) {
      Blob* logical_blob = nullptr;
      std::unique_ptr<OnDemandHostBlob> on_demand_host_logical_blob;
      if (ref_accessor.host_blob()->shape() == logical_blob_shape) {
        logical_blob = ref_accessor.host_blob();
      } else {
        on_demand_host_logical_blob.reset(new OnDemandHostBlob(logical_blob_shape, data_type));
        logical_blob = on_demand_host_logical_blob->blob();
      }
      std::mt19937 random_seed_gen(original_variable_conf.random_seed());
      InitializeWithConfUtil::SwitchInitializeWithConf(SwitchCase(data_type),
                                                       original_variable_conf.initializer(),
                                                       random_seed_gen(), logical_blob);
      if (on_demand_host_logical_blob) {
        HostSliceCopy(ref_accessor.host_blob(), slice, on_demand_host_logical_blob->blob(),
                      TensorSliceView(logical_blob_shape));
      }
    } else if (original_variable_conf.has_initialize_with_snapshot()) {
      const auto& snapshot_conf = original_variable_conf.initialize_with_snapshot();
      const std::string key = snapshot_conf.has_key() ? snapshot_conf.key() : var_lbn;
      const SnapshotReader reader(snapshot_conf.path());
      reader.Read(key, logical_blob_shape, slice, ref_accessor.host_blob());
    } else {
      UNIMPLEMENTED();
    }
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelInitV2Conf, ModelInitV2Kernel);

template<DeviceType device_type>
class ModelLoadV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadV2Kernel);
  ModelLoadV2Kernel() = default;
  ~ModelLoadV2Kernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelLoadV2OpConf& conf = this->op_conf().model_load_v2_conf();
    const Blob* path = BnInOp2Blob("path");
    Blob* ref = BnInOp2Blob("ref");
    const VariableOpConf& original_variable_conf = conf.original_variable_conf();
    const Shape logical_blob_shape(original_variable_conf.shape());
    const std::string& var_lbn =
        GenLogicalBlobName(conf.variable_op_name(), original_variable_conf.out());
    const TensorSliceView slice = GetPartSlice(this->kernel_conf());
    AutoSyncBlobAccessor<device_type> ref_accessor(ctx.device_ctx, ref, false, true);
    const std::string snapshot_path = SyncReadStringFromBlob<device_type>(ctx.device_ctx, path);
    SnapshotReader reader(snapshot_path);
    reader.Read(var_lbn, logical_blob_shape, slice, ref_accessor.host_blob());
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelLoadV2Conf, ModelLoadV2Kernel);

template<DeviceType device_type>
class ModelSaveV2Kernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2Kernel);
  ModelSaveV2Kernel() = default;
  ~ModelSaveV2Kernel() override = default;

 private:
  void Forward(const KernelCtx& ctx,
               std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const ModelSaveV2OpConf& conf = this->op_conf().model_save_v2_conf();
    const Blob* path_blob = BnInOp2Blob("path");
    Blob* in_blob = BnInOp2Blob("in");
    const ParallelContext& parallel_ctx = this->kernel_conf().model_io_v2_conf().parallel_ctx();
    const VariableOpConf& original_variable_conf = conf.original_variable_conf();
    const Shape logical_blob_shape(original_variable_conf.shape());
    const bool is_broadcast = logical_blob_shape == in_blob->shape();
    const DataType data_type = original_variable_conf.data_type();
    if (is_broadcast && parallel_ctx.parallel_id() != 0) { return; }
    const std::string snapshot_path =
        SyncReadStringFromBlob<device_type>(ctx.device_ctx, path_blob);
    AutoSyncBlobAccessor<device_type> in_accessor(ctx.device_ctx, in_blob, true, false);
    SnapshotWriter writer(snapshot_path);
    const std::string var_lbn =
        GenLogicalBlobName(conf.variable_op_name(), original_variable_conf.out());
    const std::string key = is_broadcast ? var_lbn : GetTmpPartKey(var_lbn, parallel_ctx);
    writer.Write(key, in_accessor.host_blob());
    if (!is_broadcast) {
      const int64_t parallel_num = parallel_ctx.parallel_num();
      Global<CtrlClient>::Get()->Barrier(snapshot_path + "-" + var_lbn, parallel_num);
      if (parallel_ctx.parallel_id() != 0) { return; }
      TensorSliceView total_slice(logical_blob_shape);
      OnDemandHostBlob total_blob(logical_blob_shape, data_type);
      SnapshotReader reader(snapshot_path);
      FOR_RANGE(int64_t, i, 0, parallel_num) {
        const TensorSliceView part_slice = GetPartSlice(this->kernel_conf(), i);
        const std::string part_key = GetTmpPartKey(var_lbn, i, parallel_num);
        OnDemandHostBlob part_blob(part_slice.shape(), data_type);
        reader.Read(part_key, part_blob.blob());
        HostSliceCopy(total_blob.blob(), total_slice, part_blob.blob(), part_slice);
        SnapshotFS()->RecursivelyDeleteDir(Dirname(JoinPath(snapshot_path, part_key)));
      }
      writer.Write(var_lbn, total_blob.blob());
    }
  }
};

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kModelSaveV2Conf, ModelSaveV2Kernel);

}  // namespace oneflow
