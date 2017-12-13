#ifndef ONEFLOW_CORE_JOB_MOCK_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_MOCK_JOB_DESC_H_

#include "oneflow/core/common/test_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace test {

class MockJobDesc : public JobDesc {
 public:
  MockJobDesc() = default;
  MOCK_METHOD0(SizeOfOneDataId, size_t());
  MOCK_METHOD0(DefaultDataType, DataType());
};

void InitJobDescSingleton(MockJobDesc* mock_job_desc) {
  *(JobDesc::SingletonPPtr()) = mock_job_desc;
}

}  // namespace test

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_MOCK_JOB_DESC_H_
