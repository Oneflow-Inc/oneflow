#ifndef final
#define final
#endif 

#include <gmock/gmock.h>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

template<typename T>
class MockJobDesc : public JobDesc {
 public:
  MockJobDesc() = default;
  MOCK_METHOD0(SizeOfOneDataId, int32_t());
  MOCK_METHOD0(DefaultDataType, DataType());
};

template<typename T>
void InitJobDescSingleton(MockJobDesc<T>& mock_job_desc) {
  EXPECT_CALL(mock_job_desc, SizeOfOneDataId())
      .WillRepeatedly(testing::Return(8));
  EXPECT_CALL(mock_job_desc, DefaultDataType())
      .WillRepeatedly(testing::Return(GetDataType<T>::val));
  *(JobDesc::SingletonPPtr()) = &mock_job_desc;
}

}  // namespace oneflow
