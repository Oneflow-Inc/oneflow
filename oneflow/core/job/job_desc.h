#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  ~JobDesc() = default;

  OF_SINGLETON(JobDesc);

  void ToProto(JobDescProto*) const;

  // Common
  const JobConf& job_conf() const { return job_conf_; }
  const DLNetConf& dlnet_conf() const { return dlnet_conf_; }
  const Resource& resource() const { return resource_; }
  const Placement& placement() const { return placement_; }
  const std::string& MdLoadSnapshotPath();
  DataType DefaultDataType() const { return job_conf_.default_data_type(); }
  size_t SizeOfOneDataId() const;
  int64_t TotalMachineNum() const { return resource_.machine().size(); }
  DeviceType GetDeviceType() const { return resource_.device_type(); }
  int32_t PersistenceWorkerNum() const;
  int32_t BoxingWorkerNum() const;
  int32_t CommNetWorkerNum() const;
  bool IsTrain() const { return job_conf_.has_train_conf(); }
  bool IsPredict() const { return job_conf_.has_predict_conf(); }

  // Train conf
  const std::string& MdSaveSnapshotsPath() const;
  int32_t NumOfBatchesInSnapshot() const;
  int32_t NumOfPiecesInBatch() const;
  int32_t Staleness() const;
  int64_t TotalBatchNum() const;
  const FillConf* DefaultFillConf() const;
  int32_t PieceNumOfRecordLoss() const;
  int32_t SinglePieceSize() const { return job_conf_.single_piece_size(); }
  int32_t ParallelPieceSize() const;
  int32_t BatchSize() const;

 private:
  JobDesc(const JobConf&);
  JobDesc(const JobDescProto&);

  JobConf job_conf_;
  DLNetConf dlnet_conf_;
  Resource resource_;
  Placement placement_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
