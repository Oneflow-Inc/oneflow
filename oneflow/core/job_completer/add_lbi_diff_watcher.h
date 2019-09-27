#ifndef ONEFLOW_CORE_JOB_COMPLETER_ADD_LBI_DIFF_WATCHER_H_
#define ONEFLOW_CORE_JOB_COMPLETER_ADD_LBI_DIFF_WATCHER_H_

#include "oneflow/core/job/job_builder.h"

namespace oneflow {

void AddLbiDiffWatherOpConfs(const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi,
                             JobBuilder* job_builder);
}

#endif  // ONEFLOW_CORE_JOB_COMPLETER_ADD_LBI_DIFF_WATCHER_H_
