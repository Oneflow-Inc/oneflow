#include "oneflow/core/kernel/piece_slice_v2_kernel_util.h"

namespace oneflow {

namespace {

void CheckSameStaticShape(std::vector<Blob*>& blobs) {
  Shape shape = blobs.at(0)->static_shape();
  FOR_RANGE(size_t, i, 1, blobs.size()) { CHECK_EQ(shape, blobs.at(i)->static_shape()); }
}

void CheckSameStaticShape(const std::vector<const Blob*>& blobs) {
  Shape shape = blobs.at(0)->static_shape();
  FOR_RANGE(size_t, i, 1, blobs.size()) { CHECK_EQ(shape, blobs.at(i)->static_shape()); }
}

void CheckSameDynamicShape(std::vector<Blob*>& blobs) {
  Shape shape = blobs.at(0)->shape();
  FOR_RANGE(size_t, i, 1, blobs.size()) { CHECK_EQ(shape, blobs.at(i)->shape()); }
}

void CheckSameDynamicShape(const std::vector<const Blob*>& blobs) {
  Shape shape = blobs.at(0)->shape();
  FOR_RANGE(size_t, i, 1, blobs.size()) { CHECK_EQ(shape, blobs.at(i)->shape()); }
}

}  // namespace

template<DeviceType device_type, typename T>
void PieceSliceV2KernelUtil<device_type, T>::PieceSlice(DeviceCtx* ctx, const Blob* in_blob,
                                                        std::vector<Blob*>& out_blobs) {
  size_t instance_byte_size = 0;
  if (in_blob->has_dim1_valid_num_field()) {
    CheckSameStaticShape(out_blobs);
    instance_byte_size = out_blobs.at(0)->static_shape().elem_cnt() * sizeof(T);
  } else {
    CheckSameDynamicShape(out_blobs);
    instance_byte_size = out_blobs.at(0)->shape().elem_cnt() * sizeof(T);
  }
  FOR_RANGE(size_t, i, 0, out_blobs.size()) {
    const char* src = in_blob->dptr<char>() + instance_byte_size * i;
    char* dst = out_blobs.at(i)->mut_dptr<char>();
    Memcpy<device_type>(ctx, dst, src, instance_byte_size);
  }
}

template<DeviceType device_type, typename T>
void PieceSliceV2KernelUtil<device_type, T>::InstanceStack(DeviceCtx* ctx,
                                                           const std::vector<const Blob*>& in_blobs,
                                                           Blob* out_blob) {
  size_t instance_byte_size = 0;
  if (out_blob->has_dim1_valid_num_field()) {
    CheckSameStaticShape(in_blobs);
    instance_byte_size = in_blobs.at(0)->static_shape().elem_cnt() * sizeof(T);
  } else {
    CheckSameDynamicShape(in_blobs);
    instance_byte_size = in_blobs.at(0)->shape().elem_cnt() * sizeof(T);
  }
  FOR_RANGE(size_t, i, 0, in_blobs.size()) {
    const char* src = in_blobs.at(i)->dptr<char>();
    char* dst = out_blob->mut_dptr<char>() + instance_byte_size * i;
    Memcpy<device_type>(ctx, dst, src, instance_byte_size);
  }
}

template<DeviceType device_type, typename T>
void PieceSliceV2KernelUtil<device_type, T>::SliceInstanceShape(const Blob* in_blob,
                                                                std::vector<Blob*>& out_blobs) {
  CheckSameStaticShape(out_blobs);
  if (out_blobs.at(0)->static_shape().NumAxes() < 2) { return; }
  const bool uncontiguous_varing_in =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_in = in_blob->has_instance_shape_field();
  if (contiguous_varing_in) {
    CHECK(!uncontiguous_varing_in);
    const std::vector<int64_t>& dim_vec = in_blob->instance_shape().dim_vec();
    FOR_RANGE(size_t, i, 0, out_blobs.size()) {
      out_blobs.at(i)->set_instance_shape(
          Shape(std::vector<int64_t>(dim_vec.begin() + 1, dim_vec.end())));
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void PieceSliceV2KernelUtil<device_type, T>::StackInstanceShape(
    const std::vector<const Blob*>& in_blobs, Blob* out_blob) {
  FOR_RANGE(size_t, i, 0, in_blobs.size()) {
    CHECK(in_blobs.at(i)->has_instance_shape_field());
    CHECK(!(in_blobs.at(i)->has_dim1_valid_num_field()
            || in_blobs.at(i)->has_dim2_valid_num_field()));
  }
  const Shape shape = in_blobs.at(0)->shape();
  FOR_RANGE(size_t, i, 1, in_blobs.size()) { CHECK_EQ(shape, in_blobs.at(i)->shape()); }
  CHECK(out_blob->has_instance_shape_field());
  out_blob->set_instance_shape(shape);
}

#define INSTANTIATE_PIECE_SLICE_V2_KERNEL_CPU_UTIL(type_cpp, type_proto) \
  template struct PieceSliceV2KernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_PIECE_SLICE_V2_KERNEL_CPU_UTIL, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_PIECE_SLICE_V2_KERNEL_CPU_UTIL

#define INSTANTIATE_PIECE_SLICE_V2_KERNEL_GPU_UTIL(type_cpp, type_proto) \
  template struct PieceSliceV2KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_PIECE_SLICE_V2_KERNEL_GPU_UTIL, ARITHMETIC_DATA_TYPE_SEQ)
#undef INSTANTIATE_PIECE_SLICE_V2_KERNEL_GPU_UTIL

}  // namespace oneflow