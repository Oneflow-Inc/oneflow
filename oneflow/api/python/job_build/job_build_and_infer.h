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
#ifndef ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_H_
#define ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_H_

#include <utility>
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/api/python/job_build/job_build_and_infer_helper.h"

std::shared_ptr<oneflow::cfg::ErrorProto> JobBuildAndInferCtx_Open(const std::string& job_name) {
  return oneflow::JobBuildAndInferCtx_Open(job_name).GetDataAndErrorProto();
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_GetCurrentJobName() {
  return oneflow::JobBuildAndInferCtx_GetCurrentJobName().GetDataAndErrorProto(std::string(""));
}

std::shared_ptr<oneflow::cfg::ErrorProto> JobBuildAndInferCtx_Close() {
  return oneflow::JobBuildAndInferCtx_Close().GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_CheckJob() {
  return oneflow::CurJobBuildAndInferCtx_CheckJob().GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_SetJobConf(
    const std::string& serialized_job_conf) {
  return oneflow::CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf).GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_SetTrainConf(
    const std::string& serialized_train_conf) {
  return oneflow::CurJobBuildAndInferCtx_SetTrainConf(serialized_train_conf).GetDataAndErrorProto();
}

std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_Complete() {
  return oneflow::CurJobBuildAndInferCtx_Complete().GetDataAndErrorProto();
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> CurJobBuildAndInferCtx_HasJobConf() {
  return oneflow::CurJobBuildAndInferCtx_HasJobConf().GetDataAndErrorProto(false);
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
CurJobBuildAndInferCtx_AddAndInferMirroredOp(const std::string& serialized_op_conf) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferMirroredOp(serialized_op_conf)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
CurJobBuildAndInferCtx_AddAndInferConsistentOp(const std::string& serialized_op_conf) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferConsistentOp(serialized_op_conf)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
             job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(
    const std::string& lbi_uuid_pair) {
  return oneflow::CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_uuid_pair)
      .GetDataAndErrorProto();
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(const std::string& job_name,
                                                     const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_GetDataType(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetDataType(job_name, lbn).GetDataAndErrorProto(0LL);
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_IsDynamic(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_IsDynamic(job_name, lbn).GetDataAndErrorProto(false);
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_DisableBoxing(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_DisableBoxing(job_name, lbn).GetDataAndErrorProto(false);
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_IsTensorList(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_IsTensorList(job_name, lbn).GetDataAndErrorProto(false);
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_GetBatchAxis(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_GetSplitAxisFromProducerView(const std::string& job_name,
                                                 const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(const std::string& job_name,
                                                              const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::shared_ptr<oneflow::cfg::ErrorProto> CurJobBuildAndInferCtx_AddLossLogicalBlobName(
    const std::string& lbn) {
  return oneflow::CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn).GetDataAndErrorProto();
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>> JobBuildAndInferCtx_IsMirroredBlob(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn).GetDataAndErrorProto(false);
}

std::pair<int, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn)
      .GetDataAndErrorProto(0);
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(const std::string& job_name,
                                                    const std::string& lbn, int index) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape(const std::string& job_name,
                                                                 const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<long long, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetDataType(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetDataType(job_name, lbn)
      .GetDataAndErrorProto(0LL);
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobIsDynamic(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn)
      .GetDataAndErrorProto(false);
}

std::pair<bool, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobIsTensorList(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn)
      .GetDataAndErrorProto(0LL);
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetBatchAxis(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

std::pair<std::string, std::shared_ptr<oneflow::cfg::ErrorProto>>
JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(const std::string& job_name,
                                                             const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(job_name, lbn)
      .GetDataAndErrorProto(std::string(""));
}

#endif  // ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_H_
