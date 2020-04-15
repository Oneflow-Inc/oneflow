#include "oneflow/customized/utils/pool_util.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

Params3D::Params3D(const int32_t dim, const Shape& x_shape, const std::string& data_format,
                   const std::string& padding, const std::vector<int32_t>& pool_size,
                   const std::vector<int32_t>& strides) {
  x_3d_ = {GetInDim(x_shape, data_format, 0, dim), GetInDim(x_shape, data_format, 1, dim),
           GetInDim(x_shape, data_format, 2, dim)};
  Get3DOutputSize(x_3d_, pool_size, strides, padding, &y_3d_, &padding_before_, &padding_after_);
  if (data_format == "channels_first") {
    channel_num_ = x_shape.At(1);
  } else if (data_format == "channels_last") {
    channel_num_ = x_shape.At(x_shape.NumAxes() - 1);
  } else {
    UNIMPLEMENTED();
  }
  batch_num_ = x_shape.At(0);
}

Shape Params3D::GetYShape() {
  DimVector out_shape;
  if (dim_ == 1) {
    out_shape = {y_3d_.at(2)};
  } else if (dim_ == 2) {
    out_shape = {y_3d_.at(1), y_3d_.at(2)};
  } else if (dim_ == 3) {
    out_shape = {y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)};
  } else {
    UNIMPLEMENTED();
  }
  if (data_format_ == "channels_first") {
    out_shape.insert(out_shape.begin(), channel_num_);
  } else if (data_format_ == "channels_last") {
    out_shape.insert(out_shape.end(), channel_num_);
  } else {
    UNIMPLEMENTED();
  }
  out_shape.insert(out_shape.begin(), batch_num_);
  return Shape(out_shape);
}

}  // namespace oneflow
