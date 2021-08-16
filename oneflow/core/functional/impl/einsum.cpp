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

#include "oneflow/core/common/global.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/scalar.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class EinSumFunctor {
 public:
  EinSumFunctor() {}
  Maybe<Tensor> operator()(const std::string& equation,
                           const TensorTuple& inputs) const {
    CHECK_OR_RETURN(inputs.size() > 0) << "Has at least one input tensor";

    // Code used to identify ELLIPSIS ("...")
    constexpr int ELLIPSIS = '.';

    // Find arrow (->) to split equation into lhs and rhs
    const auto arrow_pos = equation.find("->");
    const auto lhs = equation.substr(0, arrow_pos);

    const auto num_ops = inputs.size();

    // Convert labels for input operands into an index in [0, 25] and store
    // them in op_labels for each operand along with ELLIPSIS if present.
    std::vector<std::vector<int>> op_labels(num_ops);
    bool found_ell = false;
    std::size_t curr_op = 0;
    for (auto i = decltype(lhs.length()){0}; i < lhs.length(); ++i) {
      switch (lhs[i]) {
        case ' ':
          // Ignore spaces
          break;

        case '.':
          CHECK_OR_RETURN(
              // Only one ellipsis per operand can be given
              !found_ell)
              << "einsum() found \'.\' for operand " << curr_op
              << " for which an ellipsis was already found";
          CHECK_OR_RETURN(
              // Ensure it's a valid ellipsis
              i + 2 < lhs.length() && lhs[++i] == '.' && lhs[++i] == '.')
              << "einsum() found \'.\' for operand " << curr_op
              << " that is not part of any ellipsis";
          op_labels[curr_op].push_back(ELLIPSIS);
          found_ell = true;
          break;

        case ',':
          // Move onto next operand
          ++curr_op;
          CHECK_OR_RETURN(curr_op < num_ops)
              << "einsum() fewer operands were provided than specified in the "
                 "equation";
          found_ell = false;
          break;

        default:
          // Parse label
          CHECK_OR_RETURN(lhs[i] >= 'a' && lhs[i] <= 'z')
              << "einsum() operand subscript must be in range [a, z] but found "
              << lhs[i] << " for operand " << curr_op;
          // Convert label to index in [0, 25] and store
          op_labels[curr_op].push_back(lhs[i] - 'a');
      }
    }

    CHECK_OR_RETURN(curr_op == num_ops - 1)
        << "einsum() more operands were provided than specified in the "
           "equation";

    // Labels must be within [a, z].
    constexpr int TOTAL_LABELS = 'z' - 'a' + 1;
    std::vector<int> label_count(TOTAL_LABELS, 0);

    // The maximum number of dimensions covered by any ellipsis, needed when
    // unsqueezing missing dimensions from operands to permute and broadcast
    int64_t ell_num_dim = 0;

    // Compute label frequency and number of dimensions covered by ellipsis
    // We do this after parsing labels to make it more readable and simpler
    // to compute the number of dimensions covered by ellipsis.
    for (auto i = 0; i < num_ops; i++) {
      const auto operand = inputs[i];
      const auto labels = op_labels[i];
      const int ndims = operand->ndim();
      int nlabels = labels.size();
      bool has_ellipsis = false;

      for (const auto& label : labels) {
        if (label == ELLIPSIS) {
          --nlabels;
          has_ellipsis = true;
          ell_num_dim = std::max(ell_num_dim, ndims - nlabels);
        } else {
          ++label_count[label];
        }
      }
      if (has_ellipsis) {
        CHECK_OR_RETURN(nlabels <= ndims)
            << "einsum() the number of subscripts in the equation (" << nlabels
            << ") is more than the number of dimensions";
      } else {
        CHECK_OR_RETURN(nlabels == ndims)
            << ") does not match the number of dimensions (" << ndims
            << ") for operand " << i << " and no ellipsis wa given)";
      }
    }

    // We want to align the dimensions of every input tensor to have
    // shape out_dims + sum_dims. For this, we create a mapping of label
    // to index into the permuted shape.
    std::vector<int> label_perm_index(TOTAL_LABELS, -1);

    // Current index in the permuted shape
    int perm_index = 0;

    // Start index of ellipsis dimensions in the permuted shape
    int ell_index = 0;
    found_ell = false;

    if (arrow_pos == std::string::npos) {
      // Implicit output is ellipsis (...) + labels seen only once
      perm_index = ell_num_dim;
      found_ell = true;
      for (auto label = 0; label < TOTAL_LABELS; label++) {
        if (label_count[label] == 1) {
          label_perm_index[label] = perm_index++;
        }
      }
    } else {
      // Parse explicit output
      const auto rhs = equation.substr(arrow_pos + 2);
      for (auto i = decltype(rhs.length()){0}; i < rhs.length(); ++i) {
        switch (rhs[i]) {
          case ' ':
            // Ignore spaces
            break;

          case '.':
            CHECK_OR_RETURN(
                // There can only be one ellipsis in the output
                !found_ell)
                << "einsum() found \'.\' for output but an ellipsis (...) was "
                   "already found";
            CHECK_OR_RETURN(
                // Ensure ellipsis is correct
                i + 2 < rhs.length() && rhs[++i] == '.' && rhs[++i] == '.')
            "einsum() found \'.\' for output that is not part of any "
            "ellipsis (...)";
            ell_index = perm_index;
            perm_index += ell_num_dim;
            found_ell = true;
            break;

          default:
            CHECK_OR_RETURN(
                // Labels must be in [a, z]
                rhs[i] >= 'a' && rhs[i] <= 'z')
                << "einsum() subscripts must be in range [a, z] but found "
                << rhs[i] << " for the output";
            const auto label = rhs[i] - 'a';
            CHECK_OR_RETURN(
                // Ensure label appeared at least once for some input operand
                // and at most once for the output
                label_count[label] > 0 && label_perm_index[label] == -1)
                << "einsum() output subscript " << rhs[i]
                << (label_perm_index[label] > -1
                        ? " appears more than once in the output"
                        : " does not appear in the equation for any input "
                          "operand");
            label_perm_index[label] = perm_index++;
        }
      }
    }

    // Save output size before adding contraction dims (dims to sum out)
    const int out_size = perm_index;

    // If ellipsis is not part of the output, add to contraction dimensions
    if (!found_ell) {
      ell_index = perm_index;
      perm_index += ell_num_dim;
    }

    // Add contraction labels (labels not present in output)
    for (auto label = 0; label < TOTAL_LABELS; label++) {
      if (label_count[label] > 0 && label_perm_index[label] == -1) {
        label_perm_index[label] = perm_index++;
      }
    }

    // Here we unsqueeze missing dimensions to make all operands have the same
    // number of dimensions. We take diagonals for repeated labels within the
    // same operand. Finally we permute the operands to align dimensions as
    // per the perm_out_index we computed above.
    TensorTuple permuted_operands;
    for (auto i = 0; i < num_ops; i++) {
      std::vector<int> perm_shape(perm_index, -1);
      std::vector<int> label_dim(TOTAL_LABELS, -1);
      std::shared_ptr<Tensor> operand = inputs[i];
      const auto labels = op_labels[i];
      const auto original_sizes = operand->shape()->dim_vec();

      std::size_t j = 0;
      for (const auto& label : labels) {
        if (label == ELLIPSIS) {
          // Add missing dimensions covered by the ellipsis
          const int64_t num_missing_dim =
              ell_num_dim - (original_sizes.size() - labels.size() + 1);
          for (auto k = 0; k < num_missing_dim; k++) {
            operand = JUST(ExpandDims(operand, j));
          }
          for (auto k = 0; k < ell_num_dim; k++) {
            perm_shape[ell_index + k] = j++;
          }
        } else if (label_dim[label] != -1) {
          // Repeated label, take diagonal
          const auto dim = label_dim[label];
          CHECK_OR_RETURN(operand->dim(j) == operand->dim(dim))
              << "einsum() subscript " << char(label + 'a')
              << " is repeated for operand " << i
              << " but the sizes don't match, " << operand->dim(j)
              << " != " << operand->dim(dim);

          // TODO: implement diagonal(operand, bias, dim1 ,dim2, dst_dim)
          // operand = JUST(diagonal(operand, 0, j, dim, 0))
        } else {
          // Lookup output index for label
          label_dim[label] = j;
          perm_shape[label_perm_index[label]] = j++;
        }
      }

      // Add dimensions for missing labels
      for (auto index : perm_shape) {
        if (index == -1) {
          operand = JUST(ExpandDims(operand, -1));
          // ? whether support -1 as location
          index = j++;
        }
      }
      // TODO: implement permute(operand, dst_dim_list: ls[i]=j, s.t. new
      // tensor's dim_i is original tensor's dim_j])

      // permuted_operands.push_back(JUST(permute(operand, perm_shape)));
      permuted_operands.push_back(JUST(Transpose(operand, perm_shape)));
    }

    // Check if operands broadcast and keep track of last operand with
    // dimension size != 1 for optimizing reductions
    std::vector<std::size_t> dim_last_op(perm_index, 0);
    bool has_zero_size_dim = false;
    for (auto dim = 0; dim < perm_index; dim++) {
      auto broadcast_size = permuted_operands[0]->dim(dim);
      for (auto i = 1; i < num_ops; i++) {
        const auto dim_size = permuted_operands[i]->dim(dim);
        if (broadcast_size != dim_size && broadcast_size != 1 &&
            dim_size != 1) {
          std::ostringstream msg;
          msg << "einsum() operands do not broadcast with remapped shapes "
                 "[original->remapped]:";
          for (auto j = 0; j < num_ops; j++) {
            msg << " " << inputs[j]->dim(dim) << "->"
                << permuted_operands[j]->dim(dim);
          }
          CHECK_OR_RETURN(false) << msg.str();
        }
        if (dim_size != 1) {
          broadcast_size = dim_size;
          dim_last_op[dim] = i;
        }
      }
      has_zero_size_dim |= broadcast_size == 0;
    }

    // Compute result

    // std::shared_ptr<Tensor> result = Copy(permuted_operands[0], device_type,
    // device_id);
    std::shared_ptr<Tensor> result;

    // Fast path for when an operand has zero sized dim
    if (has_zero_size_dim) {
      std::vector<int64_t> out_shape(out_size);
      for (auto i = 0; i < out_size; i++) {
        out_shape[i] = permuted_operands[dim_last_op[i]]->dim(i);
      }
      return std::shared_ptr<Tensor>();
      // TODO: need function supporting construct zeros tensor with shape
    }

    // Sum out or squeeze dimensions that are size 1 for all later operands
    int dim = out_size;
    for (int i = dim; i < perm_index; ++i, ++dim) {
      if (dim_last_op[i] == 0) {
        if (result->dim(dim) == 1) {
          result = JUST(ExpandDims(result, dim--));
        } else {
          ReduceSum(result, {dim--}, false);
        }
      }
    }

    for (auto i = 1; i < num_ops; i++) {
      auto operand = permuted_operands[i];
      std::vector<int> sum_dims;

      // Sum out or squeeze dimensions that are size 1 for all later operands
      dim = out_size;
      for (int j = dim; j < perm_index; ++j, ++dim) {
        if (dim_last_op[j] < i) {
          operand = JUST(ExpandDims(operand, dim));
          --dim;
        } else if (dim_last_op[j] == i) {
          if (result->dim(dim) == 1) {
            operand = JUST(ReduceSum(operand, dim, false));
            result = JUST(ExpandDims(result, dim, false));
            --dim;
          } else {
            sum_dims.push_back(dim);
          }
        }
      }

      // Multiply tensors and sum out dimensions in sum_dims
      if (sum_dims.empty()) {
        result = JUST(Multiply(result, operand));
      } else if (sum_dims.size() == result->shape()->NumAxes()) {
        result = JUST(MatMul(Flatten(result, 0, -1), Flatten(operand, 0, -1),
                              false, true));
      } else {
        result = JUST(ReduceSum(Multiply(result, operand), sum_dims, false));
      }
    }

    return result;
  }
}

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::EinSumFunctor>("EinSum");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
