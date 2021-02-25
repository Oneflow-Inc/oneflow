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
#ifndef ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_API_H_
#define ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_API_H_

#include <utility>
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

inline void JobBuildAndInferCtx_Open(const std::string& job_name) {
  return oneflow::JobBuildAndInferCtx_Open(job_name).GetOrThrow();
}

inline std::string JobBuildAndInferCtx_GetCurrentJobName() {
  return oneflow::JobBuildAndInferCtx_GetCurrentJobName().GetOrThrow();
}

inline void JobBuildAndInferCtx_Close() {
  return oneflow::JobBuildAndInferCtx_Close().GetOrThrow();
}

inline void CurJobBuildAndInferCtx_CheckJob() {
  return oneflow::CurJobBuildAndInferCtx_CheckJob().GetOrThrow();
}

inline void CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  return oneflow::CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf).GetOrThrow();
}

inline void CurJobBuildAndInferCtx_SetTrainConf(const std::string& serialized_train_conf) {
  return oneflow::CurJobBuildAndInferCtx_SetTrainConf(serialized_train_conf).GetOrThrow();
}

inline void CurJobBuildAndInferCtx_Complete() {
  return oneflow::CurJobBuildAndInferCtx_Complete().GetOrThrow();
}

inline void CurJobBuildAndInferCtx_Rebuild() {
  return oneflow::CurJobBuildAndInferCtx_Rebuild().GetOrThrow();
}

inline bool CurJobBuildAndInferCtx_HasJobConf() {
  return oneflow::CurJobBuildAndInferCtx_HasJobConf().GetOrThrow();
}

inline std::string CurJobBuildAndInferCtx_AddAndInferMirroredOp(
    const std::string& serialized_op_conf) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferMirroredOp(serialized_op_conf).GetOrThrow();
}

inline std::string CurJobBuildAndInferCtx_AddAndInferConsistentOp(
    const std::string& serialized_op_conf) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferConsistentOp(serialized_op_conf).GetOrThrow();
}

inline void CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(const std::string& lbi_uuid_pair) {
  return oneflow::CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_uuid_pair).GetOrThrow();
}

inline std::string JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(const std::string& job_name,
                                                                        const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn).GetOrThrow();
}

inline long long JobBuildAndInferCtx_GetDataType(const std::string& job_name,
                                                 const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetDataType(job_name, lbn).GetOrThrow();
}

inline bool JobBuildAndInferCtx_IsDynamic(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_IsDynamic(job_name, lbn).GetOrThrow();
}

inline bool JobBuildAndInferCtx_DisableBoxing(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_DisableBoxing(job_name, lbn).GetOrThrow();
}

inline bool JobBuildAndInferCtx_IsTensorList(const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_IsTensorList(job_name, lbn).GetOrThrow();
}

inline std::string JobBuildAndInferCtx_GetSplitAxisFromProducerView(const std::string& job_name,
                                                                    const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn).GetOrThrow();
}

inline std::string JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(job_name, lbn)
      .GetOrThrow();
}

inline void CurJobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& lbn) {
  return oneflow::CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn).GetOrThrow();
}

inline bool JobBuildAndInferCtx_IsMirroredBlob(const std::string& job_name,
                                               const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn).GetOrThrow();
}

inline int JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(const std::string& job_name,
                                                        const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn).GetOrThrow();
}

inline std::string JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(const std::string& job_name,
                                                                       const std::string& lbn,
                                                                       int index) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index).GetOrThrow();
}

inline void JobBuildAndInferCtx_CheckLbnValidAndExist(const std::string& job_name,
                                                      const std::string& lbn) {
  return oneflow::JobBuildAndInferCtx_CheckLbnValidAndExist(job_name, lbn).GetOrThrow();
}

inline std::string JobBuildAndInferCtx_GetOpBlobLbn(const std::string& job_name,
                                                    const std::string& op_name,
                                                    const std::string bn_in_op) {
  return oneflow::JobBuildAndInferCtx_GetOpBlobLbn(job_name, op_name, bn_in_op).GetOrThrow();
}

#endif  // ONEFLOW_API_PYTHON_JOB_BUILD_JOB_BUILD_AND_INFER_API_H_
