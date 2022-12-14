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
#include <utility>
#include <vector>
#include "oneflow/core/graph/op_graph.h"
#include "OneFlow/OneFlowRoundTrip.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/job/job_ir.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

namespace {

template<IRPassType>
std::string IRPassTypeName();

template<>
std::string IRPassTypeName<kBeforeAD>() {
  return "before_ad";
}

template<>
std::string IRPassTypeName<kAfterAD>() {
  return "after_ad";
}

template<IRPassType>
bool IsLastIRPassForIRPassType();

template<>
bool IsLastIRPassForIRPassType<kBeforeAD>() {
  return false;
}

template<>
bool IsLastIRPassForIRPassType<kAfterAD>() {
  return true;
}

template<IRPassType ir_pass_type>
class RoundTripOneFlowJobWrapper : public mlir::oneflow::RoundTripOneFlowJobWrapperInterface {
 public:
  explicit RoundTripOneFlowJobWrapper(::oneflow::Job* job)
      : job_(job), op_graph_(*job), job_builder_(job), is_updated_(false) {}

  const Job* job() const override { return job_; }

  bool IsLastIRPass() const override { return IsLastIRPassForIRPassType<ir_pass_type>(); }

  void UpdateJob(::oneflow::Job* new_job) override {
    CHECK(is_updated_ == false);
    job_->Swap(new_job);
    is_updated_ = true;
  }
  void DumpLog(const std::string& filename, const std::string& content) override {
    if (IsInDebugMode()) {
      TeePersistentLogStream::Create(JoinPath(LogDir(), filename))->Write(content);
    }
  }

  const ::oneflow::ParallelConf& ParallelConf4OpName(const std::string& op_name) const override {
    return job_builder_.ParallelConf4OpName(op_name).GetOrThrow();
  }
  const ::oneflow::OperatorConf& OpConf4OpName(const std::string& op_name) const override {
    return job_builder_.OpConf4OpName(op_name).GetOrThrow();
  }
  std::pair<std::vector<std::string>, std::vector<std::string>> InputBns4OpName(
      const std::string& op_name) const override {
    auto node = op_graph_.OpNode4OpName(op_name);
    std::vector<std::string> input_bns{};
    std::vector<std::string> input_lbns{};
    for (auto e : node->in_edges()) {
      for (const auto& lbi_ibn_pair : e->lbi2ibns()) {
        for (const auto& ibn : lbi_ibn_pair.second) {
          input_bns.push_back(ibn);
          input_lbns.push_back(GenLogicalBlobName(lbi_ibn_pair.first));
        }
      }
    }
    return std::make_pair(input_bns, input_lbns);
  }

  std::vector<std::string> OutputLbns4OpName(const std::string& op_name) const override {
    std::unordered_set<std::string> ret{};
    auto node = op_graph_.OpNode4OpName(op_name);
    for (auto e : node->out_edges()) {
      for (const auto& lbi : e->lbis()) { ret.insert(GenLogicalBlobName(lbi)); }
    }
    return {ret.begin(), ret.end()};
  }

  std::string ReplaceInputLbnInOpCustomizedConf(::oneflow::OperatorConf* op_conf,
                                                const std::string& ibn,
                                                const std::string& new_val) const override {
    return ::oneflow::ReplaceInputLbnInOpCustomizedConf(op_conf, ibn, new_val);
  }

  void QueryLogicalBlob(
      const std::string& lbn,
      std::function<void(const int64_t* shape_begin, const int64_t* shape_end, DataType dt)> cb)
      const override {
    LogicalBlobId lbi = GenLogicalBlobId(lbn);
    auto& blob_desc = op_graph_.GetLogicalBlobDesc(lbi);
    cb(blob_desc.shape().dim_vec().begin(), blob_desc.shape().dim_vec().end(),
       blob_desc.data_type());
  }

  void TopoForEachOpConf(
      std::function<void(const ::oneflow::OperatorConf*)> Handler) const override {
    op_graph_.TopoForEachNodeWithCtrlEdge(
        [&](OpNode* op_node) { Handler(&op_node->op().op_conf()); });
  }

  std::string LogDir() {
    return JoinPath("ir_pass", IRPassTypeName<ir_pass_type>(), job_->job_conf().job_name());
  }

 private:
  Job* job_;
  const OpGraph op_graph_;
  JobBuilder job_builder_;
  bool is_updated_;
};

}  // namespace

template<IRPassType ir_pass_type>
bool IRRoundTrip<ir_pass_type>::IsEnabled(const JobPassCtx& ctx) const {
  return ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_ROUND_TRIP", false);
}

void SortJob(Job& job) {
  auto* ops = job.mutable_net()->mutable_op();
  std::sort(ops->begin(), ops->end(),
            [](const oneflow::OperatorConf& l, const oneflow::OperatorConf& r) {
              return l.name() < r.name();
            });
}

template<IRPassType ir_pass_type>
Maybe<void> IRRoundTrip<ir_pass_type>::Apply(Job* job, JobPassCtx* ctx) const {
  if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  Job job_before{};
  job_before.CopyFrom(*job);
  RoundTripOneFlowJobWrapper<ir_pass_type> w(job);
  SortJob(job_before);
  if (IsInDebugMode()) {
    TeePersistentLogStream::Create(JoinPath(w.LogDir(), "job_before_ir_round_trip.prototxt"))
        ->Write(job_before);
  }
  mlir::oneflow::RoundTripOneFlowJob(w, [](::oneflow::Job* job, std::string& reason) {
    // TODO: It is not clear how to define if extra boxing is introduced
    TODO();
    return true;
  });
  if (IsInDebugMode()) {
    Job job_after{};
    job_after.CopyFrom(*job);
    SortJob(job_after);
    TeePersistentLogStream::Create(JoinPath(w.LogDir(), "job_after_ir_round_trip.prototxt"))
        ->Write(job_after);
  }
  return Maybe<void>::Ok();
}

template class IRRoundTrip<kBeforeAD>;
template class IRRoundTrip<kAfterAD>;

Maybe<std::string> ConvertJobToTosaIR(Job* job) {
  RoundTripOneFlowJobWrapper<kBeforeAD> job_wrapper(job);
  return ::mlir::oneflow::ConvertJobToTosaIR(job_wrapper);
}

Maybe<void> SaveJobToIR(Job* job, const std::string& path) {
  // TODO: check path is valid dir
  if (IsInDebugMode()) { TeePersistentLogStream::Create("saved_job")->Write(*job); }
  RoundTripOneFlowJobWrapper<kBeforeAD> job_wrapper(job);
  ::mlir::oneflow::SaveJobToIR(job_wrapper, path);
  return Maybe<void>::Ok();
}

Maybe<std::string> ConvertJobToIR(Job* job) {
  if (IsInDebugMode()) { TeePersistentLogStream::Create("saved_job")->Write(*job); }
  RoundTripOneFlowJobWrapper<kBeforeAD> job_wrapper(job);
  return ::mlir::oneflow::ConvertJobToIR(job_wrapper);
}

Maybe<void> LoadJobFromIR(Job* job, const std::string& path) {
  job->Clear();
  RoundTripOneFlowJobWrapper<kBeforeAD> job_wrapper(job);
  ::mlir::oneflow::LoadJobFromIR(job_wrapper, path);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
