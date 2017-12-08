#ifndef final
#define final
#endif

#include <gmock/gmock.h>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

class MockJobDesc : public JobDesc {
 public:
  MockJobDesc() = default;
  MOCK_METHOD0(SizeOfOneDataId, size_t());
  MOCK_METHOD0(DefaultDataType, DataType());
};

void InitJobDescSingleton(MockJobDesc& mock_job_desc, size_t size_of_one_dataid,
                          DataType data_type) {
  EXPECT_CALL(mock_job_desc, SizeOfOneDataId())
      .WillRepeatedly(testing::Return(size_of_one_dataid));
  EXPECT_CALL(mock_job_desc, DefaultDataType())
      .WillRepeatedly(testing::Return(data_type));
  *(JobDesc::SingletonPPtr()) = &mock_job_desc;
}

}  // namespace oneflow
