#ifndef ONEFLOW_CORE_JOB_MODEL_IO_JOB_
#define ONEFLOW_CORE_JOB_MODEL_IO_JOB_

#include "oneflow/core/job/compiler.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"

namespace oneflow {

void FilterVariableOps(std::vector<Job>&, HashMap<std::string, OperatorConf>*);
void MakeModelInitJob(const std::string&, Job*, const HashMap<std::string, OperatorConf>&,
                      const HashMap<std::string, ParallelBlobConf>&);
void MakeModelLoadJob(const std::string&, Job*, const HashMap<std::string, OperatorConf>&,
                      const HashMap<std::string, ParallelBlobConf>&);
void MakeModelSaveJob(const std::string&, Job*, const HashMap<std::string, OperatorConf>&,
                      const HashMap<std::string, ParallelBlobConf>&);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MODEL_IO_JOB_
