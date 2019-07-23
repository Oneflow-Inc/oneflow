#include "oneflow/core/kernel/util/host_arithemetic_interface.h"

namespace oneflow {

namespace {

void ComputeOffset(const int32_t num_axes, const int64_t* shape, const int32_t* permutation,
                   std::vector<int64_t>& offset) {
  offset.resize(num_axes);
  std::vector<int64_t> buff(num_axes);
  int64_t cur_offset = 1;
  for (int32_t i = num_axes - 1; i >= 0; --i) {
    buff[i] = cur_offset;
    cur_offset *= shape[i];
  }
  for (int32_t i = 0; i < num_axes; ++i) { offset[permutation[i]] = buff[i]; }
}

void IncreaseIndex(const int64_t* shape, std::vector<int64_t>& index) {
  for (int32_t i = index.size() - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= shape[i]) {
      index[i] -= shape[i];
    } else {
      break;
    }
  }
}

template<typename T>
void TransposeImpl(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                   const Shape& y_shape, const PbRf<int32_t>& permutation, const int64_t elem_cnt,
                   const T* x, T* y) {
  int64_t block_size = 1;
  int32_t shared_idxs_num = 0;
  for (int32_t i = num_axis - 1; i >= 0 && permutation[i] == i; --i) {
    block_size *= y_shape.At(i);
    ++shared_idxs_num;
  }
  if (num_axis < 2 || shared_idxs_num == num_axis) {
    memcpy(y, x, elem_cnt * sizeof(T));
    return;
  }
  int32_t trans_axis = num_axis - shared_idxs_num;
  std::vector<int64_t> x_to_y_offset;
  ComputeOffset(trans_axis, y_shape.dim_vec().data(), permutation.data(), x_to_y_offset);
  std::vector<int64_t> x_index_digits(trans_axis, 0);
  int64_t num_blocks = elem_cnt / block_size;
  FOR_RANGE(int64_t, x_idx, 0, num_blocks) {
    int64_t y_idx = std::inner_product(x_to_y_offset.cbegin(), x_to_y_offset.cend(),
                                       x_index_digits.cbegin(), 0);
    if (block_size == 1) {
      y[y_idx] = x[x_idx];
    } else {
      memcpy(y + block_size * y_idx, x + block_size * x_idx, block_size * sizeof(T));
    }
    IncreaseIndex(x_shape.dim_vec().data(), x_index_digits);
  }
}

template<typename T>
void ConstantInitializer(const T& value, Blob* blob) {
  T* dptr = blob->mut_dptr<T>();
  const int64_t elem_cnt = blob->shape().elem_cnt();
  CHECK(elem_cnt);
  for (int64_t i = 0; i < elem_cnt; ++i) { dptr[i] = value; }
}

}  // namespace

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const Shape& x_shape, const Shape& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const float* x, float* y) {
  TransposeImpl<float>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const Shape& x_shape, const Shape& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const double* x,
                                                double* y) {
  TransposeImpl<double>(ctx, num_axis, x_shape, y_shape, permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kCPU>::InitializeWithConstConf(
    DeviceCtx* ctx, const ConstantInitializerConf& initializer_conf, uint32_t random_seed,
    Blob* blob) {
  DataType dtype = blob->data_type();
  if (dtype == DataType::kFloat) {
    ConstantInitializer<float>(initializer_conf.value(), blob);
  } else if (dtype == DataType::kDouble) {
    ConstantInitializer<double>(static_cast<double>(initializer_conf.value()), blob);
  } else if (dtype == DataType::kFloat16) {
    ConstantInitializer<float16>(static_cast<float16>(initializer_conf.value()), blob);
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
