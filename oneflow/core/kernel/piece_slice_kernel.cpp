#ifndef ONEFLOW_CORE_KERNEL_PIECE_SLICE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PIECE_SLICE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PieceSliceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceKernel);
  PieceSliceKernel() = default;
  ~PieceSliceKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* in_blob = BnInOp2Blob("in");
    const int32_t out_size = this->op_conf().piece_slice_conf().out_size();
    CHECK(in_blob->use_lod());
    // num_lod_level() should be equal to lod_desc().at(0).size() - 1
    CHECK_EQ(in_blob->num_lod_level(), 1);
    // lod_desc() is a 2-level vector, represent offset, start with 0
    // recursive_sequence_lengths() is an another 2-level vector, represent length
    // Think about better names for them
    CHECK_EQ(out_size, in_blob->lod_desc().at(0).size() - 1);
    const int32_t dense_byte_size = in_blob->dense_shape().elem_cnt() * sizeof(T);
    FOR_RANGE(size_t, i, 0, out_size) {
      const char* src = in_blob->dptr<char>() + in_blob->lod_desc().at(0).at(i) * dense_byte_size;
      char* dst = BnInOp2Blob("out_" + std::to_string(i))->mut_dptr<char>();
      Memcpy<device_type>(ctx, dst, src,
                          in_blob->recursive_sequence_lengths.at(0).at(i) * dense_byte_size);
    }
  }
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const int32_t out_size = this->op_conf().piece_slice_conf().out_size();
    FOR_RANGE(size_t, i, 0, out_size) {
      const Blob* in_blob = BnInOp2Blob("in");
      auto dim_vec = in_blob->dense_shape().dim_vec();
      dim_vec.insert(dev_vec.begin(), in_blob->recursive_sequence_lengths().at(0).at(i));
      BnInOp2Blob("out_" + std::to_string(i))->set_dense_shape(Shape(dim_vec));
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PIECE_SLICE_KERNEL_H_
