#include <map>
#include "oneflow/core/device/hccl_util.h"
namespace oneflow{

std::map<enum DataType, HcclDataType> hcclDataType = {
    {DataType::kChar, HCCL_DATA_TYPE_INT8},
    {DataType::kFloat, HCCL_DATA_TYPE_FP32},
    {DataType::kInt32, HCCL_DATA_TYPE_INT32},
    {DataType::kFloat16, HCCL_DATA_TYPE_FP16},
    {DataType::kInt16, HCCL_DATA_TYPE_INT16},
    {DataType::kInt64, HCCL_DATA_TYPE_INT64},
};

}