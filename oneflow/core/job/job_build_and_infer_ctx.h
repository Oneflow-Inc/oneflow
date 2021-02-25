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
#ifndef ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
#define ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

class JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtx);
  JobBuildAndInferCtx(Job* job, int64_t job_id);
  virtual ~JobBuildAndInferCtx() = default;

  Maybe<void> SetJobConf(const JobConfigProto& job_conf);
  Maybe<void> AddLbiAndDiffWatcherUuidPair(const LbiAndDiffWatcherUuidPair& lbi_uuid_pair);
  Maybe<OpAttribute> AddAndInferConsistentOp(const OperatorConf& op_conf);
  Maybe<OpAttribute> AddAndInferMirroredOp(const OperatorConf& op_conf);
  Maybe<void> AddLossLogicalBlobName(const std::string& lbn);
  Maybe<void> SetTrainConf(const TrainConf& train_conf);

  bool HasJobConf() const;
  Maybe<Shape> GetStaticShape(const std::string& lbn) const;
  Maybe<DataType> GetDataType(const std::string& lbn) const;
  Maybe<bool> IsDynamic(const std::string& lbn) const;
  Maybe<bool> DisableBoxing(const std::string& lbn) const;
  Maybe<bool> IsTensorList(const std::string& lbn) const;
  Maybe<OptInt64> GetSplitAxisFromProducerView(const std::string& lbn) const;
  Maybe<const ParallelDesc*> GetParallelDescFromProducerView(const std::string& lbn) const;

  bool IsMirroredBlob(const std::string& lbn) const;
  Maybe<int> MirroredBlobGetNumSubLbi(const std::string& lbn) const;
  Maybe<const LogicalBlobId*> MirroredBlobGetSubLbi(const std::string& lbn, int index) const;

  Maybe<Shape> MirroredBlobGetStaticShape(const std::string& lbn_with_hint) const;
  Maybe<DataType> MirroredBlobGetDataType(const std::string& lbn_with_hint) const;
  Maybe<bool> MirroredBlobIsDynamic(const std::string& lbn_with_hint) const;
  Maybe<bool> MirroredBlobIsTensorList(const std::string& lbn_with_hint) const;
  Maybe<OptInt64> MirroredBlobGetSplitAxisFromProducerView(const std::string& lbn_with_hint) const;
  Maybe<const ParallelDesc*> MirroredBlobGetParallelDescFromProducerView(
      const std::string& lbn_with_hint) const;

  const Job& job() const;
  Maybe<void> CheckJob() const;
  std::string GetJobStructureGraphJson(const std::string& job_name) const;
  Maybe<void> CheckLbnValidAndExist(const std::string& lbn) const;
  Maybe<void> Rebuild();
  Maybe<std::string> GetOpBlobLbn(const std::string& op_name, const std::string& bn_in_op) const;

  virtual Maybe<void> Complete() = 0;

 protected:
  virtual Maybe<void> CheckAllInputsWithSameParallelNum(const Operator& op,
                                                        int32_t parallel_num) const = 0;
  virtual std::string GetMirroredOpName(const std::string& op_name, int64_t parallel_id) const = 0;
  virtual int64_t SizeOfSubConsistentOpList(int64_t parallel_num) const = 0;
  virtual ParallelConf GetMirroredOpParallelConf(const ParallelDesc&,
                                                 int64_t parallel_id) const = 0;
  virtual bool GetIsMirroredParallelView() const = 0;
  virtual Maybe<LogicalBlobId> FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
      int64_t scope_symbol_id, const LogicalBlobId& lbn) = 0;

  Job* mut_job() const { return job_; }
  int64_t job_id() const { return job_id_; }
  const HashMap<LogicalBlobId, std::vector<LogicalBlobId>>& mirrored_lbi2sub_lbis() const {
    return mirrored_lbi2sub_lbis_;
  }
  HashMap<LogicalBlobId, std::vector<LogicalBlobId>>* mut_mirrored_lbi2sub_lbis() {
    return &mirrored_lbi2sub_lbis_;
  }
  Maybe<const ParallelDesc*> ParallelDesc4Lbi(const LogicalBlobId& lbi) const;
  HashMap<LogicalBlobId, LogicalBlobId>* mut_consistent_lbi2mirrored_lbi() {
    return &consistent_lbi2mirrored_lbi_;
  }
  Maybe<const SbpParallel*> SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  bool IsVariableLbi(const LogicalBlobId& lbi) const;
  Maybe<Operator*> Op4OpName(const std::string& op_name) const;
  Maybe<OpAttribute> AddAndInferOp(const OperatorConf& op_conf, const ParallelConf& parallel_conf,
                                   const JobDesc* job_desc, bool is_mirrored_parallel_view);

 private:
  Maybe<ParallelConf> InferOpParallelConf(
      const Operator& op, const ParallelConf& origin_parallel_conf,
      const HashMap<std::string, bool>& ibn2disable_boxing) const;
  Maybe<void> AddOpNameParallelConf2Placement(const std::string& op_name,
                                              const ParallelConf& parallel_conf);
  void InitIbn2DisableBoxing(const Operator& op, HashMap<std::string, bool>* ibn2disable_boxing);
  void UpdateLbi2DisableBoxing(const Operator& op,
                               const HashMap<std::string, bool>& ibn2disable_boxing);
  Maybe<void> AddLbiParallelConf2BlobPlacement(
      const Operator* op, std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn);
  Maybe<OperatorConf> DecodeLbiHintAndReturnNewOpConf(
      const Operator& op, SbpSignature* sbp_sig_conf,
      HashMap<std::string, bool>* ibn2disable_boxing) const;
  void AddOpAndUpdateJobParallelViewConf(const OperatorConf& operator_conf,
                                         const SbpSignature& sbp_signature,
                                         bool is_mirrored_parallel_view) const;
  Maybe<void> InferMirroredSignature(Operator*, bool is_mirrored_parallel_view_conf,
                                     const ParallelDesc&);
  Maybe<void> InferOpOutSbpParallel(Operator*, const SbpSignature&, const ParallelDesc&);
  Maybe<void> GenOpProducedEmptyLogicalBlobDesc(Operator* op);
  Maybe<void> CheckOpBlobSplitability(Operator*, int64_t parallel_num);
  Maybe<void> CheckPlacement() const;
  Maybe<void> CheckJobConf() const;
  Maybe<LogicalBlobId> GetMirroredLbi(const std::string& lbn_with_hint) const;
  bool HasAnyMirroredBlobInput(const Operator& op) const;
  Maybe<void> CheckAllInputsConvertableToMirroredBlob(const Operator& op) const;
  Maybe<void> AddLossConsistentBlobName(const std::string& lbn);
  Maybe<void> AddLossMirroredBlobName(const std::string& lbn);
  Maybe<const LogicalBlobId*> GetSubLbi(int64_t scope_symbol_id, const LogicalBlobId& lbi,
                                        int32_t index);
  Maybe<bool> AllInputsBroadcastParallel(const Operator& op) const;
  void InferBlobBackwardSignature(Operator* op);
  void InferBlobBackwardSignature(const Operator& op,
                                  std::function<bool(const LogicalBlobId&)>* IsLbiBackwardUsed);

  Job* job_;
  int64_t job_id_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  HashMap<LogicalBlobId, SbpParallel> lbi2sbp_parallel_from_producer_view_;
  HashMap<LogicalBlobId, ParallelDesc> lbi2parallel_desc_from_producer_view_;
  HashMap<LogicalBlobId, bool> lbi2disable_boxing_;
  HashMap<std::string, std::shared_ptr<Operator>> op_name2op_;
  HashMap<ParallelDesc, PlacementGroup*> parallel_desc2placement_group_;
  HashMap<ParallelDesc, BlobPlacementGroup*> parallel_desc2blob_placement_group_;
  HashMap<LogicalBlobId, LogicalBlobId> consistent_lbi2mirrored_lbi_;
  HashMap<LogicalBlobId, std::vector<LogicalBlobId>> mirrored_lbi2sub_lbis_;
  HashMap<LogicalBlobId, ParallelDesc> mirrored_lbi2parallel_desc_;
  HashMap<LogicalBlobId, SbpParallel> mirrored_lbi2sbp_parallel_;
  bool is_job_conf_frozen_;
  bool has_job_conf_;
  HashMap<std::string, bool> op_name2ancestors_need_no_grad_;
};

class LazyJobBuildAndInferCtx : public JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyJobBuildAndInferCtx);
  LazyJobBuildAndInferCtx(Job* job, int64_t job_id) : JobBuildAndInferCtx(job, job_id) {}
  virtual ~LazyJobBuildAndInferCtx() = default;

 private:
  Maybe<void> Complete() override;
  Maybe<void> CheckAllInputsWithSameParallelNum(const Operator& op,
                                                int32_t parallel_num) const override;
  std::string GetMirroredOpName(const std::string& op_name, int64_t parallel_id) const override;
  int64_t SizeOfSubConsistentOpList(int64_t parallel_num) const override { return parallel_num; }
  ParallelConf GetMirroredOpParallelConf(const ParallelDesc&, int64_t parallel_id) const override;
  bool GetIsMirroredParallelView() const override { return false; }
  Maybe<LogicalBlobId> FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
      int64_t scope_symbol_id, const LogicalBlobId& lbn) override;
};

class EagerJobBuildAndInferCtx : public JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerJobBuildAndInferCtx);
  EagerJobBuildAndInferCtx(Job* job, int64_t job_id) : JobBuildAndInferCtx(job, job_id) {}
  virtual ~EagerJobBuildAndInferCtx() = default;

 private:
  Maybe<void> Complete() override;
  Maybe<void> CheckAllInputsWithSameParallelNum(const Operator& op,
                                                int32_t parallel_num) const override;
  std::string GetMirroredOpName(const std::string& op_name, int64_t parallel_id) const override;
  int64_t SizeOfSubConsistentOpList(int64_t parallel_num) const override { return 1; }
  ParallelConf GetMirroredOpParallelConf(const ParallelDesc&, int64_t parallel_id) const override;
  bool GetIsMirroredParallelView() const override { return true; }
  Maybe<LogicalBlobId> FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
      int64_t scope_symbol_id, const LogicalBlobId& lbn) override;

  HashSet<std::string> executed_op_names_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
