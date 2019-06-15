#include "oneflow/core/kernel/local_scatter_nd_update_kernel.h"

namespace oneflow {

namespace {

template<typename K>
int64_t GetOffsetInNdarray(const std::vector<int64_t>& shape_dim_vec, const K* index_ptr,
                           const int64_t index_dim) {
  int64_t offset = 0;
  FOR_RANGE(int64_t, i, 0, index_dim) {
    const int64_t stride = std::accumulate(shape_dim_vec.begin() + 1 + i, shape_dim_vec.end(), 1,
                                           std::multiplies<int64_t>());
    offset += index_ptr[i] * stride;
  }
  return offset;
}

}  // namespace

template<DeviceType device_type, typename T, typename K>
const PbMessage& LocalScatterNdUpdateKernel<device_type, T, K>::GetCustomizedOpConf() const {
  return this->op_conf().local_scatter_nd_update_conf();
}

template<DeviceType device_type, typename T, typename K>
void LocalScatterNdUpdateKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* indices_blob = BnInOp2Blob("indices");
  const Blob* updates_blob = BnInOp2Blob("updates");
  if (this->op_conf().device_type() == DeviceType::kGPU) {}
  Blob* out_blob = BnInOp2Blob("out");

  const auto indices_dim_vec = indices_blob->shape().dim_vec();
  const int64_t indices_dim = indices_dim_vec.back();
  const auto updates_dim_vec = updates_blob->shape().dim_vec();
  FOR_RANGE(size_t, i, 0, indices_blob->shape().NumAxes() - 1) {
    CHECK_EQ(indices_dim_vec.at(i), updates_dim_vec.at(i));
  }
  CHECK_LE(indices_dim, in_blob->shape().NumAxes());
  FOR_RANGE(int64_t, i, indices_dim, in_blob->shape().NumAxes()) {
    CHECK_EQ(in_blob->shape().At(i), updates_dim_vec.at(i));
  }
  const int64_t num_updates = std::accumulate(indices_dim_vec.begin(), indices_dim_vec.end() - 1, 1,
                                              std::multiplies<int64_t>());
  const int64_t block_size = std::accumulate(updates_dim_vec.begin() + indices_dim - 1,
                                             updates_dim_vec.end(), 1, std::multiplies<int64_t>());
  int64_t* shape_ptr = (this->op_conf().device_type() == DeviceType::kGPU)
                           ? BnInOp2Blob("shape")->mut_dptr<int64_t>()
                           : nullptr;
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  LocalScatterNdUpdateKernelUtil<device_type, T, K>::Forward(
      ctx.device_ctx, shape_ptr, indices_blob, updates_blob, num_updates, block_size, out_blob);
}

template<DeviceType device_type, typename T, typename K>
void LocalScatterNdUpdateKernel<device_type, T, K>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  const Blob* indices_blob = BnInOp2Blob("indices");
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  Blob* updates_diff_blob = BnInOp2Blob(GenDiffBn("updates"));

  const auto indices_dim_vec = indices_blob->shape().dim_vec();
  const int64_t indices_dim = indices_dim_vec.back();
  const auto updates_diff_dim_vec = updates_diff_blob->shape().dim_vec();
  const auto out_diff_dim_vec = out_diff_blob->shape().dim_vec();
  const int64_t num_updates = std::accumulate(indices_dim_vec.begin(), indices_dim_vec.end() - 1, 1,
                                              std::multiplies<int64_t>());
  const int64_t block_size =
      std::accumulate(updates_diff_dim_vec.begin() + indices_dim - 1, updates_diff_dim_vec.end(), 1,
                      std::multiplies<int64_t>());
  int64_t* shape_ptr = (this->op_conf().device_type() == DeviceType::kGPU)
                           ? BnInOp2Blob("shape")->mut_dptr<int64_t>()
                           : nullptr;
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  LocalScatterNdUpdateKernelUtil<device_type, T, K>::Backward(
      ctx.device_ctx, out_diff_blob, shape_ptr, indices_blob, num_updates, block_size,
      updates_diff_blob, in_diff_blob);
}

template<DeviceType device_type, typename T, typename K>
void LocalScatterNdUpdateKernel<device_type, T, K>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDim0ValidNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T, typename K>
void LocalScatterNdUpdateKernel<device_type, T, K>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyInstanceShapeFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<typename T, typename K>
struct LocalScatterNdUpdateKernelUtil<DeviceType::kCPU, T, K> {
  static void Forward(DeviceCtx* ctx, int64_t* shape_ptr, const Blob* indices_blob,
                      const Blob* updates_blob, const int64_t num_updates, const int64_t block_size,
                      Blob* out_blob) {
    CHECK(shape_ptr == nullptr);
    const int64_t index_dim = indices_blob->shape().dim_vec().back();
    FOR_RANGE(int64_t, i, 0, num_updates) {
      const K* index_ptr = indices_blob->dptr<K>() + i * index_dim;
      T* to = out_blob->mut_dptr<T>()
              + GetOffsetInNdarray(out_blob->shape().dim_vec(), index_ptr, index_dim) * block_size;
      memset(to, 0, block_size * sizeof(T));
    }
    FOR_RANGE(int64_t, i, 0, num_updates) {
      const T* from = updates_blob->dptr<T>() + i * block_size;
      const K* index_ptr = indices_blob->dptr<K>() + i * index_dim;
      T* to = out_blob->mut_dptr<T>()
              + GetOffsetInNdarray(out_blob->shape().dim_vec(), index_ptr, index_dim) * block_size;
      std::transform(from, from + block_size, to, to, std::plus<T>());
    }
  }
  static void Backward(DeviceCtx* ctx, const Blob* out_diff_blob, int64_t* shape_ptr,
                       const Blob* indices_blob, const int64_t num_updates,
                       const int64_t block_size, Blob* updates_diff_blob, Blob* in_diff_blob) {
    CHECK(shape_ptr == nullptr);
    const int64_t index_dim = indices_blob->shape().dim_vec().back();
    FOR_RANGE(int64_t, i, 0, num_updates) {
      const K* index_ptr = indices_blob->dptr<K>() + i * index_dim;
      const int64_t offset =
          GetOffsetInNdarray(in_diff_blob->shape().dim_vec(), index_ptr, index_dim);
      const T* from = out_diff_blob->dptr<T>() + offset * block_size;
      T* to = updates_diff_blob->mut_dptr<T>() + i * block_size;
      std::copy(from, from + block_size, to);
      memset(in_diff_blob->mut_dptr<T>() + offset, 0, block_size * sizeof(T));
    }
  }
};

namespace {

Kernel* CreateLocalScatterNdUpdateKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define LOCAL_SCATTER_ND_UPDATE_KERNEL_ENTRY(device_type, value_type_pair, indices_type_pair) \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(value_type_pair),                                \
              OF_PP_PAIR_SECOND(indices_type_pair)),                                          \
   []() {                                                                                     \
     return new LocalScatterNdUpdateKernel<device_type, OF_PP_PAIR_FIRST(value_type_pair),    \
                                           OF_PP_PAIR_FIRST(indices_type_pair)>();            \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(LOCAL_SCATTER_ND_UPDATE_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type(),
                                kernel_conf.local_scatter_nd_update_conf().indices_type(),
                                kernel_conf.local_scatter_nd_update_conf().value_type()))();
}

}  // namespace

REGISTER_KERNEL_CREATOR(OperatorConf::kLocalScatterNdUpdateConf, CreateLocalScatterNdUpdateKernel);

#define MAKE_ENTRY(value_type_pair, indices_type_pair) \
  template struct LocalScatterNdUpdateKernelUtil<      \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(value_type_pair), OF_PP_PAIR_FIRST(indices_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
