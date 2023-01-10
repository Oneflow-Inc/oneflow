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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/boxing/slice_boxing_util.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/user/kernels/communicate_util.h"

namespace oneflow {

namespace private_details {

Maybe<one::Tensor> PreprocessInputTensor4SliceBoxing(const std::shared_ptr<one::Tensor>& tensor,
                                                     const std::string& log_prefix) {
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  if (IsSendAndRecvRegistered(tensor_placement->device_type())) { return tensor; }

  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  Symbol<ParallelDesc> new_placement = JUST(ReplaceDeviceType(tensor_placement, DeviceType::kCPU));

  const auto& boxing_interpreter =
      JUST(Singleton<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
          tensor_nd_sbp, tensor_nd_sbp, tensor_placement, new_placement, *tensor->shape()));
  Singleton<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()), log_prefix);
  return JUST(boxing_interpreter->Interpret(tensor, tensor_nd_sbp, tensor_nd_sbp, tensor_placement,
                                            new_placement));
}

Maybe<one::Tensor> PostprocessOutputTensor4SliceBoxing(const std::shared_ptr<one::Tensor>& tensor,
                                                       Symbol<PlacedNdSbp> placed_nd_sbp,
                                                       const std::string& log_prefix) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_nd_sbp == placed_nd_sbp->nd_sbp())
      << Error::RuntimeError()
      << "Compute slice boxing failed.  Please submit an issue in "
         "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
         "possible";
  CHECK_OR_RETURN(tensor_placement->EqualsIgnoringDeviceType(*placed_nd_sbp->placement()))
      << Error::RuntimeError()
      << "Compute slice boxing failed. Please submit an issue in "
         "`https://github.com/Oneflow-Inc/oneflow/issues` and we will fix it as soon as "
         "possible";

  if (JUST(tensor->parallel_desc()) == placed_nd_sbp->placement()) { return tensor; }
  const auto& boxing_interpreter =
      JUST(Singleton<EagerBoxingInterpreterManager>::Get()->GetEagerBoxingInterpreter(
          placed_nd_sbp->nd_sbp(), placed_nd_sbp->nd_sbp(), JUST(tensor->parallel_desc()),
          placed_nd_sbp->placement(), *tensor->shape()));
  Singleton<const EagerBoxingLogger>::Get()->Log(
      *JUST(boxing_interpreter->boxing_interpreter_status()), log_prefix);
  return JUST(boxing_interpreter->Interpret(tensor, placed_nd_sbp->nd_sbp(),
                                            placed_nd_sbp->nd_sbp(), JUST(tensor->parallel_desc()),
                                            placed_nd_sbp->placement()));
}

const std::string& LogPrefix4EagerSliceBoxingType(EagerSliceBoxingType boxing_type) {
  static thread_local const HashMap<EagerSliceBoxingType, std::string> boxing_type2log_prefix = {
      {EagerSliceBoxingType::kNaiveBToS, "\t\tInternal boxing of naive-b-to-s, "},
      {EagerSliceBoxingType::kNaivePToB, "\t\tInternal boxing of naive-p-to-b, "},
      {EagerSliceBoxingType::kNaivePToS, "\t\tInternal boxing of naive-p-to-s, "},
      {EagerSliceBoxingType::kNaiveSToB, "\t\tInternal boxing of naive-s-to-b, "},
      {EagerSliceBoxingType::kNaiveSToP, "\t\tInternal boxing of naive-s-to-p, "},
      {EagerSliceBoxingType::kNaiveSToS, "\t\tInternal boxing of naive-s-to-s, "}};
  return CHECK_JUST(MapAt(boxing_type2log_prefix, boxing_type));
}

}  // namespace private_details

}  // namespace oneflow
