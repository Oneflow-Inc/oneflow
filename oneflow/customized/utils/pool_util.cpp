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

Shape Params3D::GetYShape() const {
  DimVector y_dim_vec;
  if (dim_ == 1) {
    y_dim_vec = {y_3d_.at(2)};
  } else if (dim_ == 2) {
    y_dim_vec = {y_3d_.at(1), y_3d_.at(2)};
  } else if (dim_ == 3) {
    y_dim_vec = {y_3d_.at(0), y_3d_.at(1), y_3d_.at(2)};
  } else {
    UNIMPLEMENTED();
  }
  if (data_format_ == "channels_first") {
    y_dim_vec.insert(y_dim_vec.begin(), channel_num_);
  } else if (data_format_ == "channels_last") {
    y_dim_vec.insert(y_dim_vec.end(), channel_num_);
  } else {
    UNIMPLEMENTED();
  }
  y_dim_vec.insert(y_dim_vec.begin(), batch_num_);
  return Shape(y_dim_vec);
}

}  // namespace oneflow
