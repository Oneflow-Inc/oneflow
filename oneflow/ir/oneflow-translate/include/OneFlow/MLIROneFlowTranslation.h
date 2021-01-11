#ifndef ONEFLOW_MLIRONEFLOWTRANSLATION_H
#define ONEFLOW_MLIRONEFLOWTRANSLATION_H

#include "oneflow/core/job/job.pb.h"

namespace mlir {

void roundTripOneFlowJob(::oneflow::Job *job);
void registerFromOneFlowJobTranslation();

}  // namespace mlir

#endif /* ONEFLOW_MLIRONEFLOWTRANSLATION_H */
