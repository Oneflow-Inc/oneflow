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
#ifdef WITH_MLIR
#include <utility>
#include <vector>
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "oneflow/core/framework/user_op_def.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace {

class RoundTripOneFlowJobWrapper : public mlir::RoundTripOneFlowJobWrapperInterface {
 public:
  RoundTripOneFlowJobWrapper(::oneflow::Job* job)
      : job_(job), op_graph_(*job), job_builder_(job), is_updated_(false) {}

  const Job* job() const { return job_; }
  void UpdateJob(::oneflow::Job* new_job) {
    CHECK(is_updated_ == false);
    job_->Swap(new_job);
    is_updated_ = true;
  }
  void DumpLog(const std::string& filename, const std::string& content) {
    TeePersistentLogStream::Create(JoinPath(LogDir(), filename))->Write(content);
  }

  const oneflow::ParallelConf& ParallelConf4OpName(const std::string& op_name) const {
    return job_builder_.ParallelConf4OpName(op_name);
  }
  const ::oneflow::OperatorConf& OpConf4OpName(const std::string& op_name) const {
    return job_builder_.OpConf4OpName(op_name).GetOrThrow();
  }
  std::pair<std::vector<std::string>, std::vector<std::string>> InputBns4OpName(
      const std::string& op_name) const {
    auto node = op_graph_.OpNode4OpName(op_name);
    std::vector<std::string> input_bns{};
    std::vector<std::string> input_lbns{};
    for (auto e : node->in_edges()) {
      for (auto lbi_ibn_pair : e->lbi2ibns()) {
        for (auto ibn : lbi_ibn_pair.second) {
          input_bns.push_back(ibn);
          input_lbns.push_back(GenLogicalBlobName(lbi_ibn_pair.first));
        }
      }
    }
    return std::make_pair(input_bns, input_lbns);
  }

  std::vector<std::string> OutputLbns4OpName(const std::string& op_name) const {
    std::unordered_set<std::string> ret{};
    auto node = op_graph_.OpNode4OpName(op_name);
    for (auto e : node->out_edges()) {
      for (auto lbi : e->lbis()) { ret.insert(GenLogicalBlobName(lbi)); }
    }
    return {ret.begin(), ret.end()};
  }

  std::string ReplaceInputLbnInOpCustomizedConf(::oneflow::OperatorConf* op_conf,
                                                const std::string& ibn,
                                                const std::string& new_val) const {
    return ::oneflow::ReplaceInputLbnInOpCustomizedConf(op_conf, ibn, new_val);
  }

  AttrType QueryAttrType(const std::string& op_type_name, const std::string& attr_name) const {
    const user_op::OpRegistryResult* val =
        user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(op_type_name);
    CHECK(val) << " Cannot find op_type_name: " << op_type_name;
    user_op::UserOpDefWrapper op_def(val->op_def);
    CHECK(op_def.IsAttrName(attr_name)) << attr_name << " not a attr name for op: " << op_type_name;
    return op_def.GetAttrType(attr_name);
  }

  void QueryLogicalBlob(
      const std::string& lbn,
      std::function<void(const int64_t* shape_begin, const int64_t* shape_end, DataType dt)> cb)
      const {
    LogicalBlobId lbi = GenLogicalBlobId(lbn);
    auto& blob_desc = op_graph_.GetLogicalBlobDesc(lbi);
    cb(blob_desc.shape().dim_vec().begin(), blob_desc.shape().dim_vec().end(),
       blob_desc.data_type());
  }

  void TopoForEachOpConf(std::function<void(const ::oneflow::OperatorConf*)> Handler) const {
    op_graph_.TopoForEachNode([&](OpNode* op_node) { Handler(&op_node->op().op_conf()); });
  }

  std::string LogDir() { return JoinPath("ir_pass", job_->job_conf().job_name()); }

 private:
  Job* job_;
  const OpGraph op_graph_;
  JobBuilder job_builder_;
  bool is_updated_;
};

class IRRoundTrip final : public JobPass {
 public:
  IRRoundTrip() = default;
  ~IRRoundTrip() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_ROUND_TRIP", false);
  }
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    RoundTripOneFlowJobWrapper w(job);
    TeePersistentLogStream::Create(JoinPath(w.LogDir(), "job_before_ir_round_trip.prototxt"))
        ->Write(*job);
    mlir::RoundTripOneFlowJob(w, [](::oneflow::Job* job, std::string& reason) {
      // TODO: It is not clear how to define if extra boxing is introduced
      TODO();
      return true;
    });
    TeePersistentLogStream::Create(JoinPath(w.LogDir(), "job_after_ir_round_trip.prototxt"))
        ->Write(*job);
    return Maybe<void>::Ok();
  }
};

}  // namespace

REGISTER_JOB_PASS("IRRoundTrip", IRRoundTrip);

}  // namespace oneflow
#endif  // WITH_MLIR
