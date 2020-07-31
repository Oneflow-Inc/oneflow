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
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/common/util.h"
#include <json.hpp>

namespace oneflow {

Maybe<void> JobBuildAndInferCtxMgr::OpenJobBuildAndInferCtx(const std::string& job_name) {
  CHECK_OR_RETURN(!has_cur_job_) << JobBuildAndInferError::kUnknownJobBuildAndInferError
                                 << "cur job not leave before you enter this job_name:" << job_name;
  CHECK_OR_RETURN(!job_name.empty()) << JobBuildAndInferError::kJobNameEmpty;
  CHECK_OR_RETURN(job_name2infer_ctx_.find(job_name) == job_name2infer_ctx_.end())
      << JobBuildAndInferError::kJobNameExist << "job name: " << job_name << " already exist";
  int64_t job_id = job_set_.job_size();
  Job* job = job_set_.add_job();
  job->mutable_job_conf()->set_job_name(job_name);
  std::unique_ptr<JobBuildAndInferCtx> ctx(NewJobBuildAndInferCtx(job, job_id));
  job_name2infer_ctx_.emplace(job_name, std::move(ctx));
  cur_job_name_ = job_name;
  has_cur_job_ = true;
  return Maybe<void>::Ok();
}

JobBuildAndInferCtx* LazyJobBuildAndInferCtxMgr::NewJobBuildAndInferCtx(Job* job,
                                                                        int64_t job_id) const {
  return new LazyJobBuildAndInferCtx(job, job_id);
}

JobBuildAndInferCtx* EagerJobBuildAndInferCtxMgr::NewJobBuildAndInferCtx(Job* job,
                                                                         int64_t job_id) const {
  return new EagerJobBuildAndInferCtx(job, job_id);
}

Maybe<JobBuildAndInferCtx*> JobBuildAndInferCtxMgr::FindJobBuildAndInferCtx(
    const std::string& job_name) {
  CHECK_OR_RETURN(job_name2infer_ctx_.find(job_name) != job_name2infer_ctx_.end())
      << JobBuildAndInferError::kNoJobBuildAndInferCtx << "cannot find job name:" << job_name;
  return job_name2infer_ctx_.at(job_name).get();
}

Maybe<std::string> JobBuildAndInferCtxMgr::GetCurrentJobName() const {
  CHECK_OR_RETURN(has_cur_job_) << JobBuildAndInferError::kNoJobBuildAndInferCtx
                                << "current JobBuildAndInferCtx was closed, job name: "
                                << cur_job_name_;
  return cur_job_name_;
}

Maybe<void> JobBuildAndInferCtxMgr::AddLbiAndDiffWatcherUuidPair(
    const LbiAndDiffWatcherUuidPair& lbi_uuid_pair) const {
  auto* job_name2pairs =
      Global<LbiDiffWatcherInfo>::Get()->mutable_job_name2lbi_and_watcher_uuids();
  const auto& job_name = JUST(GetCurrentJobName());
  LbiAndDiffWatcherUuidPairList* pairs = &(*job_name2pairs)[*job_name];
  auto PairFoundCond = [&](const LbiAndDiffWatcherUuidPair& x) {
    return x.lbi() == lbi_uuid_pair.lbi() && x.watcher_uuid() == lbi_uuid_pair.watcher_uuid();
  };
  auto found_iter = std::find_if(pairs->lbi_and_uuid_pair().begin(),
                                 pairs->lbi_and_uuid_pair().end(), PairFoundCond);
  CHECK_OR_RETURN(found_iter == pairs->lbi_and_uuid_pair().end())
      << "diff blob has been watched. (logical_blob_name: "
      << GenLogicalBlobName(lbi_uuid_pair.lbi()) << ", job_name: " << *job_name << ")";
  *pairs->mutable_lbi_and_uuid_pair()->Add() = lbi_uuid_pair;
  return Maybe<void>::Ok();
}

Maybe<void> JobBuildAndInferCtxMgr::CloseCurrentJobBuildAndInferCtx() {
  VirtualCloseJob();
  if (!has_cur_job_) { return Maybe<void>::Ok(); }
  has_cur_job_ = false;
  const JobDesc* job_desc = Global<JobDesc>::Get();
  if (job_desc == nullptr) { return Maybe<void>::Ok(); }
  CHECK_EQ_OR_RETURN(job_desc->job_name(), cur_job_name_);
  CHECK_EQ_OR_RETURN(job_desc->job_id(), job_set_.job_size() - 1);
  Global<JobDesc>::Delete();
  return Maybe<void>::Ok();
}

std::string JobBuildAndInferCtxMgr::structure_graph() const {
  nlohmann::json json_array;
  for (const auto& pair : job_name2infer_ctx_) {
    nlohmann::json json_pair;
    json_pair["class_name"] = "Model";
    std::string tmp_json = pair.second->GetJobStructureGraphJson(pair.first);
    json_pair["config"] = nlohmann::json::parse(tmp_json);
    json_pair["backend"] = "oneflow";
    json_array.emplace_back(json_pair);
  }
  return json_array.dump();
}

void EagerJobBuildAndInferCtxMgr::VirtualCloseJob() {
  mut_job_set()->clear_job();
  clear_job_name2infer_ctx();
}

}  // namespace oneflow
