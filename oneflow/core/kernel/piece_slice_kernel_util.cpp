#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PieceSliceKernelUtil<device_type, T>::PieceSlice(DeviceCtx* ctx, const size_t ins_idx,
                                                      const size_t valid_ins_num,
                                                      const Blob* in_blob, Blob* out_blob) {
  size_t instance_byte_size = 0;
  if (in_blob->has_dim1_valid_num_field()) {
    instance_byte_size = out_blob->static_shape().elem_cnt() * sizeof(T);
  } else {
    instance_byte_size = out_blob->shape().elem_cnt() * sizeof(T);
  }
  const char* src = in_blob->dptr<char>() + instance_byte_size * ins_idx;
  char* dst = out_blob->mut_dptr<char>();
  Memcpy<device_type>(ctx, dst, src, instance_byte_size);
}

template<DeviceType device_type, typename T>
void PieceSliceKernelUtil<device_type, T>::InstanceStack(DeviceCtx* ctx, const size_t ins_idx,
                                                         const size_t valid_ins_num,
                                                         const Blob* in_blob, Blob* out_blob) {
  size_t instance_byte_size = 0;
  if (out_blob->has_dim1_valid_num_field()) {
    instance_byte_size = in_blob->static_shape().elem_cnt() * sizeof(T);
  } else {
    instance_byte_size = in_blob->shape().elem_cnt() * sizeof(T);
  }
  const char* src = in_blob->dptr<char>();
  char* dst = out_blob->mut_dptr<char>() + instance_byte_size * ins_idx;
  Memcpy<device_type>(ctx, dst, src, instance_byte_size);
}

template<DeviceType device_type, typename T>
void PieceSliceKernelUtil<device_type, T>::SliceInstanceShape(const Blob* in_blob, Blob* out_blob) {
  if (out_blob->shape().NumAxes() < 2) { return; }
  const bool uncontiguous_varing_instance =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_instance = in_blob->has_instance_shape_field();
  if (contiguous_varing_instance) {
    CHECK(!uncontiguous_varing_instance);
    const std::vector<int64_t>& dim_vec = in_blob->instance_shape().dim_vec();
    out_blob->set_instance_shape(Shape(std::vector<int64_t>(dim_vec.begin() + 1, dim_vec.end())));
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
bool PieceSliceKernelUtil<device_type, T>::StackInstanceShape(const bool is_first_instance,
                                                              const Blob* in_blob, Blob* out_blob) {
  CHECK(in_blob->has_instance_shape_field());
  CHECK(!(in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field()));
  CHECK(out_blob->has_instance_shape_field());
  if (is_first_instance) {
    out_blob->set_instance_shape(in_blob->shape());
  } else {
    const std::vector<int64_t>& dim_vec = out_blob->shape().dim_vec();
    CHECK_EQ(in_blob->shape(), Shape(std::vector<int64_t>(dim_vec.begin() + 1, dim_vec.end())));
  }
  return false;
}

#define INSTANTIATE_PIECE_SLICE_KERNEL_CPU_UTIL(type_cpp, type_proto) \
  template struct PieceSliceKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_PIECE_SLICE_KERNEL_CPU_UTIL, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_PIECE_SLICE_KERNEL_CPU_UTIL

#define INSTANTIATE_PIECE_SLICE_KERNEL_GPU_UTIL(type_cpp, type_proto) \
  template struct PieceSliceKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_PIECE_SLICE_KERNEL_GPU_UTIL, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_PIECE_SLICE_KERNEL_GPU_UTIL

}  // namespace oneflow