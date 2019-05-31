#include "oneflow/core/kernel/embedding_lookup_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void EmbeddingLookupKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("ids");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), 0,
                      sizeof(T) * out_blob->shape().elem_cnt());
  EmbeddingLookupKernelUtil<device_type, T>::Forward(ctx.device_ctx, in_blob, weight_blob,
                                                     out_blob);
}

template<DeviceType device_type, typename T>
void EmbeddingLookupKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("ids");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Blob* ids_diff_blob = BnInOp2Blob("ids_diff");
  Memcpy<device_type>(ctx.device_ctx, weight_diff_blob->mut_dptr<T>(), out_diff_blob->dptr<T>(),
                      sizeof(T) * out_diff_blob->shape().elem_cnt());
  Memcpy<device_type>(ctx.device_ctx, ids_diff_blob->mut_dptr<T>(), in_blob->dptr<T>(),
                      sizeof(T) * in_blob->shape().elem_cnt());
}

template<DeviceType device_type, typename T>
void EmbeddingLookupKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      this->GetInitializerFromPbMessage(this->op_conf().embedding_lookup_conf(),
                                        "weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("weight"));
}

template<DeviceType device_type, typename T>
void EmbeddingLookupKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t table_size = this->op_conf().embedding_lookup_conf().table_size();
  int32_t units = this->op_conf().embedding_lookup_conf().units();
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, weight_blob,
                                                "weight", table_size, units);
}

template<typename T>
class EmbeddingLookupKernelUtil<DeviceType::kCPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EmbeddingLookupKernelUtil);
  EmbeddingLookupKernelUtil() = delete;

  static void Forward(DeviceCtx* ctx, const Blob* in_blob, const Blob* weight_blob,
                      Blob* out_blob) {
    const int32_t* in_dptr = in_blob->dptr<int32_t>();
    const T* weight_dptr = weight_blob->dptr<T>();
    const int32_t units = out_blob->shape().Count(1);
    T* out_dptr = out_blob->mut_dptr<T>();

    FOR_RANGE(int32_t, n, 0, in_blob->shape().At(0)) {
      CHECK(in_dptr[n] < weight_blob->shape().At(0));
      const int32_t idx = in_dptr[n];
      if (idx == -1) { continue; }
      Memcpy<DeviceType::kCPU>(ctx, out_dptr + n * units, weight_dptr + idx * units,
                               sizeof(T) * units);
    }
  }

  static void Backward(DeviceCtx* ctx, const Blob* in_blob, const Blob* out_diff_blob,
                       Blob* ids_diff_blob, Blob* weight_diff_blob) {
    const int32_t* in_dptr = in_blob->dptr<int32_t>();
    const T* out_diff_dptr = out_diff_blob->dptr<T>();
    T* weight_diff_dptr = weight_diff_blob->mut_dptr<T>();
    T* ids_diff_dptr = ids_diff_blob->mut_dptr<T>();
    Memcpy<DeviceType::kCPU>(ctx, weight_diff_dptr, out_diff_dptr,
                             sizeof(T) * out_diff_blob->shape().elem_cnt());
    Memcpy<DeviceType::kCPU>(ctx, ids_diff_dptr, in_dptr, sizeof(T) * in_blob->shape().elem_cnt());
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kEmbeddingLookupConf, EmbeddingLookupKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
