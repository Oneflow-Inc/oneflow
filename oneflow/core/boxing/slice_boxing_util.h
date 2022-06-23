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
#ifndef ONEFLOW_CORE_BOXING_SLICE_BOXING_UTIL_H_
#define ONEFLOW_CORE_BOXING_SLICE_BOXING_UTIL_H_

#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

enum class EagerSliceBoxingType : unsigned int;

namespace private_details {

// Copy to cpu if device of input tensor is not cpu or cuda, otherwise return self
Maybe<one::Tensor> PreprocessInputTensor4SliceBoxing(const std::shared_ptr<one::Tensor>& tensor,
                                                     const std::string& log_prefix);

// Copy to corresponding device if device of output tensor is not same with that of placed_nd_sbp,
// otherwise return self
Maybe<one::Tensor> PostprocessOutputTensor4SliceBoxing(const std::shared_ptr<one::Tensor>& tensor,
                                                       Symbol<PlacedNdSbp> placed_nd_sbp,
                                                       const std::string& log_prefix);

const std::string& LogPrefix4EagerSliceBoxingType(EagerSliceBoxingType boxing_type);

}  // namespace private_details

enum class EagerSliceBoxingType : unsigned int {
  kNaiveBToS = 0,
  kNaivePToB = 1,
  kNaivePToS = 2,
  kNaiveSToB = 3,
  kNaiveSToP = 4,
  kNaiveSToS = 5
};

template<EagerSliceBoxingType boxing_type>
struct EagerSliceBoxingAutoConvert {
  template<Maybe<one::Tensor> (*func)(const std::shared_ptr<one::Tensor>&, Symbol<PlacedNdSbp>,
                                      Symbol<PlacedNdSbp>)>
  static Maybe<one::Tensor> Call(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                                 Symbol<PlacedNdSbp> out) {
    std::shared_ptr<one::Tensor> processed_in_tensor =
        JUST(private_details::PreprocessInputTensor4SliceBoxing(
            tensor, private_details::LogPrefix4EagerSliceBoxingType(boxing_type)));
    const auto& new_in =
        JUST(PlacedNdSbp::New(in->nd_sbp(), JUST(processed_in_tensor->parallel_desc())));
    Symbol<ParallelDesc> new_out_placement = JUST(ReplaceDeviceType(
        out->placement(), JUST(processed_in_tensor->parallel_desc())->device_type()));
    const auto& new_out = JUST(PlacedNdSbp::New(out->nd_sbp(), new_out_placement));
    std::shared_ptr<one::Tensor> out_tensor = JUST(func(processed_in_tensor, new_in, new_out));
    return JUST(private_details::PostprocessOutputTensor4SliceBoxing(
        out_tensor, out, private_details::LogPrefix4EagerSliceBoxingType(boxing_type)));
  }
};

#define EAGER_SLICE_BOXING_WARPPER(fn_ptr, boxing_type) \
  (&EagerSliceBoxingAutoConvert<boxing_type>::Call<fn_ptr>)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BOXING_SLICE_BOXING_UTIL_H_
