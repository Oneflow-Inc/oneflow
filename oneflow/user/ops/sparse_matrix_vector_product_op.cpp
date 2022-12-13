/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ auto SparseMatrixVectorProductOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  // no sbp strategy
  return Maybe<void>::Ok();
}

/* static */ auto SparseMatrixVectorProductOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  // obtain input shape
  const Shape& mat_rows_shape = ctx->InputShape("mat_rows", 0);
  const Shape& mat_cols_shape = ctx->InputShape("mat_cols", 0);
  const Shape& mat_values_shape = ctx->InputShape("mat_values", 0);
  const Shape& in_vec_shape = ctx->InputShape("in_vec", 0);
  const std::string format = ctx->Attr<std::string>("format");
  const int64_t num_rows = ctx->Attr<int64_t>("num_rows");

  // check attributes
  CHECK_OR_RETURN(format == "csr" || format == "csc" || format == "coo")
      << "unknown data format " << format;
  CHECK_GT_OR_RETURN(num_rows, 0) << "invalid number of rows attribute" << num_rows;

  // check dimension
  CHECK_EQ_OR_RETURN(mat_rows_shape.NumAxes(), 1)
      << "number of axes of \'mat_rows\' should be 1, yet get " << mat_rows_shape.NumAxes();
  CHECK_EQ_OR_RETURN(mat_cols_shape.NumAxes(), 1)
      << "number of axes of \'mat_cols\' should be 1, yet get " << mat_cols_shape.NumAxes();
  CHECK_EQ_OR_RETURN(mat_values_shape.NumAxes(), 1)
      << "number of axes of \'mat_values\' should be 1, yet get " << mat_values_shape.NumAxes();
  CHECK_EQ_OR_RETURN(in_vec_shape.NumAxes(), 1)
      << "number of axes of \'in_vec\' should be 1, yet get " << in_vec_shape.NumAxes();

  // check input shape
  size_t num_mat_rows = mat_rows_shape.At(0);
  size_t num_mat_cols = mat_cols_shape.At(0);
  size_t num_mat_values = mat_values_shape.At(0);
  if (format == "csr") {
    CHECK_EQ_OR_RETURN(num_mat_cols, num_mat_values)
        << "under CSR format, "
        << "the number of elements in \'mat_cols\'(" << num_mat_cols
        << ") should be equal to the one of \'mat_values\'(" << num_mat_values << ")";
    CHECK_EQ_OR_RETURN(num_mat_rows, num_rows + 1)
        << "under CSR format, "
        << "the number of elements in \'mat_rows\'(" << num_mat_rows
        << ") should be equal to the given attribute \'num_rows\'+1 (" << num_rows + 1 << ")";
  } else if (format == "csc") {
    CHECK_EQ_OR_RETURN(num_mat_rows, num_mat_values)
        << "under CSC format, "
        << "the number of elements in \'mat_rows\'(" << num_mat_rows
        << ") should be equal to the one of \'mat_values\'(" << num_mat_values << ")";
  } else if (format == "coo") {
    CHECK_EQ_OR_RETURN(num_mat_rows, num_mat_cols)
        << "under COO format, "
        << "the number of elements in \'mat_rows\'(" << num_mat_rows
        << ") should be equal to the one of \'mat_cols\'(" << num_mat_cols << ")";
    CHECK_EQ_OR_RETURN(num_mat_rows, num_mat_values)
        << "under COO format, "
        << "the number of elements in \'mat_rows\'(" << num_mat_rows
        << ") should be equal to the one of \'mat_values\'(" << num_mat_values << ")";
  }

  // set shape of the output tensor
  Shape out_vec_shape = in_vec_shape;  // borrow from in_vec_shape
  out_vec_shape.Set(0, num_rows);
  user_op::TensorDesc* out_tensor = ctx->MutOutputTensorDesc("out_vec", 0);
  out_tensor->set_shape(out_vec_shape);

  return Maybe<void>::Ok();
}

/* static */ auto SparseMatrixVectorProductOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto SparseMatrixVectorProductOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  // obtain input data types
  const user_op::TensorDesc& mat_rows_desc = ctx->InputTensorDesc("mat_rows", 0);
  const user_op::TensorDesc& mat_cols_desc = ctx->InputTensorDesc("mat_cols", 0);

  // check whether both mat_cols and mat_rows are index data type
  const DataType mat_rows_dtype = mat_rows_desc.data_type();
  const DataType mat_cols_dtype = mat_cols_desc.data_type();
  CHECK_OR_RETURN(IsIndexDataType(mat_rows_dtype))
      << Error::TypeError() << "The dtype of mat_rows must be integer, but found "
      << DataType_Name(mat_rows_dtype);
  CHECK_OR_RETURN(IsIndexDataType(mat_cols_dtype))
      << Error::TypeError() << "The dtype of mat_cols must be integer, but found "
      << DataType_Name(mat_cols_dtype);
  CHECK_EQ_OR_RETURN(mat_rows_dtype, mat_cols_dtype)
      << Error::TypeError() << "The dtype of mat_rows (" << DataType_Name(mat_rows_dtype) << ")"
      << "is not consistent with"
      << "the dtype of mat_cols (" << DataType_Name(mat_cols_dtype) << ")";

  // check data type of the value of both sparse matrix and vector
  DataType mat_values_dtype = ctx->InputDType("mat_values", 0);
  DataType in_vec_dtype = ctx->InputDType("in_vec", 0);
  CHECK_EQ_OR_RETURN(mat_values_dtype, in_vec_dtype)
      << "data type of \'mat_values\' is not consitant with \'in_vec\'";

  // set output data type
  ctx->SetOutputDType("out_vec", 0, in_vec_dtype);

  return Maybe<void>::Ok();
}

}  // namespace oneflow
