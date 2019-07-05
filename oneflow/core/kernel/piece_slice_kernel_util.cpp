#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PieceSliceKernelUtil<device_type>::PieceSlice(DeviceCtx* ctx, const size_t ins_idx,
                                                   const size_t valid_ins_num, const Blob* in_blob,
                                                   Blob* out_blob) {
  const size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
  const char* src = in_blob->dptr<char>() + out_byte_size * ins_idx;
  char* dst = out_blob->mut_dptr<char>();
  Memcpy<device_type>(ctx, dst, src, out_byte_size);
}

template<DeviceType device_type>
void PieceSliceKernelUtil<device_type>::InstanceStack(DeviceCtx* ctx, const size_t ins_idx,
                                                      const size_t valid_ins_num,
                                                      const Blob* in_blob, Blob* out_blob) {
  const size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  const char* src = in_blob->dptr<char>();
  char* dst = out_blob->mut_dptr<char>() + in_byte_size * ins_idx;
  Memcpy<device_type>(ctx, dst, src, in_byte_size);
}

template<DeviceType device_type>
void PieceSliceKernelUtil<device_type>::SliceInstanceShape(const Blob* in_blob, Blob* out_blob) {
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

template<DeviceType device_type>
bool PieceSliceKernelUtil<device_type>::StackInstanceShape(const bool is_first_instance,
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

template class PieceSliceKernelUtil<DeviceType::kCPU>;
template class PieceSliceKernelUtil<DeviceType::kGPU>;

}  // namespace oneflow