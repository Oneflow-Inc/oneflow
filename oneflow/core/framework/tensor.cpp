#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/register/blob.h"

namespace oneflow {

namespace user_op {

Tensor::Tensor(Blob* blob) {
  dptr_ = blob->ForceMutDptr();
  shape_ = blob->shape();
  blob_access_checker_ = blob->blob_access_checker();
  if (blob->ForceMutShapeView()) {
    mut_shape_.reset(new MutShapeView(*blob->ForceMutShapeView()));
  } else {
    mut_shape_.reset();
  }
  data_type_ = blob->data_type();
  mem_case_ = &(blob->mem_case());
}

void Tensor::header_access_check() { this->blob_access_checker_->CheckHeaderMutable(); }

void Tensor::body_access_check() { this->blob_access_checker_->CheckBodyMutable(); }

void Tensor::CopyWithoutData(const Tensor& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  if (rhs.mut_shape_) {
    mut_shape_.reset(new MutShapeView(*rhs.mut_shape_));
  } else {
    mut_shape_.reset();
  }
  data_type_ = rhs.data_type_;
  mem_case_ = rhs.mem_case_;
  blob_access_checker_ = rhs.blob_access_checker_;
}

Tensor& Tensor::operator=(Tensor&& rhs) {
  dptr_ = rhs.dptr_;
  shape_ = rhs.shape_;
  mut_shape_ = std::move(rhs.mut_shape_);
  data_type_ = rhs.data_type_;
  mem_case_ = rhs.mem_case_;
  blob_access_checker_ = rhs.blob_access_checker_;
  return *this;
}

}  // namespace user_op

}  // namespace oneflow
