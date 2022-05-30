#ifndef ONEFLOW_USER_OPS_NPU_COMMAND_H_
#define ONEFLOW_USER_OPS_NPU_COMMAND_H_
#ifdef WITH_NPU
#include <vector>
#include <iostream>
#include <string>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/npu/npu_stream.h"
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
namespace oneflow
{
void testblock();
static std::map<DataType, aclDataType> datatype_map ={
                {kFloat, ACL_FLOAT},
                {kFloat16, ACL_FLOAT16},
                {kInt8, ACL_INT8},
                {kInt32, ACL_INT32},
                {kUInt8, ACL_UINT8},
                {kInt16, ACL_INT16},
                {kUInt16, ACL_UINT16},
                {kUInt32, ACL_UINT32},
                {kInt64, ACL_INT64},
                {kUInt64, ACL_UINT64},
                {kDouble, ACL_DOUBLE},
                {kBool, ACL_BOOL},
                {kComplex64, ACL_COMPLEX64},
                {kComplex128, ACL_COMPLEX128},
            };   
static std::map<std::string, aclDataType> datatype_string_map ={
                {"kFloat", ACL_FLOAT},
                {"kFloat16", ACL_FLOAT16},
                {"kInt8", ACL_INT8},
                {"kInt32", ACL_INT32},
                {"kUInt8", ACL_UINT8},
                {"kInt16", ACL_INT16},
                {"kUInt16", ACL_UINT16},
                {"kUInt32", ACL_UINT32},
                {"kInt64", ACL_INT64},
                {"kUInt64", ACL_UINT64},
                {"kDouble", ACL_DOUBLE},
                {"kBool", ACL_BOOL},
                {"kComplex64", ACL_COMPLEX64},
                {"kComplex128", ACL_COMPLEX128},
            };
static std::map<std::string, aclFormat> format_map ={
                {"channels_last", ACL_FORMAT_NHWC},
                {"channels_first", ACL_FORMAT_NCHW},
                {"channels_nd", ACL_FORMAT_ND},
                {"channels_nc1hwc0",ACL_FORMAT_NC1HWC0},
            };
static std::map<std::string, std::string> attr_format_map ={
                {"channels_last", "NHWC"},
                {"channels_first", "NCHW"},
            };

#define NOT_NULLPTR_CHECK(x) if(x==nullptr) {                                           \
          std::cout<<"Get nullptr error, create input "<<#x<<" failed"<<std::endl; \
        }
aclDataType dataTypeMap(DataType type);
void PrintResult(void * out_buffers, uint32_t out_tensor_size, std::string type);
void PrintResult(void * out_buffers, uint32_t out_tensor_size, int data_len);
void PrintResult(user_op::Tensor* out);
void PrintResult(user_op::Tensor* out, std::string true_type);

void vector32To64(std::vector<int32_t>& src, std::vector<int64_t>& dst);

template<typename dtype>
dtype mulVector(std::vector<dtype> &v)
{
    return std::accumulate(v.begin(),v.end(),1,[&](dtype a, dtype b){
        return a*b;
    });
}

template<typename dtype>
dtype mulVector(const std::vector<dtype> &v)
{
    return std::accumulate(v.begin(),v.end(),1,[&](dtype a, dtype b){
        return a*b;
    });
}
#define VECTOR_PRINT(x) std::cout<<#x<<" ";\
                        for(auto& i:x) { std::cout<<i<<" ";}\
                        std::cout<<std::endl;

#define NPU_COMMAND_CHECK(x) if(!(x)) {\
                            std::cout<<"npu_command check fail "<<#x<<std::endl;\
                            return ;\
                            }
// template<typename dtype>
// struct NpuCommandSimpleTensorDesc
// {
//     dtype* data_ptr;
//     size_t count;
//     std::string format;
//     DataType data_type = kInvalidDataType;
//     NpuCommandSimpleTensorDesc(dtype* ptr, size_t cnt, std::string fmt)
//                                 :data_ptr(ptr), count(cnt), format(fmt) {}
// };
// template<>
// struct NpuCommandSimpleTensorDesc<float>
// {
//     float* data_ptr;
//     size_t count;
//     std::string format;
//     DataType data_type = kFloat;
//     NpuCommandSimpleTensorDesc(float* ptr, size_t cnt, std::string fmt)
//                                 :data_ptr(ptr), count(cnt), format(fmt) {}
// };
// template<>
// struct NpuCommandSimpleTensorDesc<float16>
// {
//     float16* data_ptr;
//     size_t count;
//     std::string format;
//     DataType data_type = kFloat16;
//     NpuCommandSimpleTensorDesc(float16* ptr, size_t cnt, std::string fmt)
//                                 :data_ptr(ptr), count(cnt), format(fmt) {}
// };

struct AclTensorWrapper
{
    AclTensorWrapper(void *ptr, aclDataType data_type, int ndims, const int64_t* dims,
                    aclFormat format, uint32_t size )
            : AclTensorWrapper(ptr, data_type, ndims, dims, format, size, nullptr, false) {}

    AclTensorWrapper(void *ptr, aclDataType data_type, int ndims, const int64_t* dims,
                    aclFormat format, uint32_t size, void* data_ptr)
            : AclTensorWrapper(ptr, data_type, ndims, dims, format, size, data_ptr, false) {
                if(data_ptr) OF_NPU_CHECK(aclrtMemcpy(tensor_ptr, tensor_size, data_ptr, tensor_size, ACL_MEMCPY_HOST_TO_DEVICE));
                //std::cout<<"Wrap Memcpy"<<std::endl;
            }

    AclTensorWrapper(void *ptr, aclDataType data_type, int ndims, const int64_t* dims,
                    aclFormat format, uint32_t size, void* data_ptr, bool isConst)
            : tensor_ptr(ptr), data_type(data_type), num_dims(ndims), 
              dims(dims), format(format), tensor_size(size),
              data_ptr(data_ptr), isConst(isConst) {
                  // fix : use NpuAllocator to malloc
                  OF_NPU_CHECK(aclrtMalloc(&tensor_ptr, tensor_size, ACL_MEM_MALLOC_NORMAL_ONLY));//dck_caution_here
                  //std::cout<<"Wrap Malloc"<<std::endl;
              }
    
    void* tensor_ptr;
    aclDataType data_type;
    int num_dims;
    const int64_t * dims;
    aclFormat format;
    uint32_t tensor_size;
    void* data_ptr;
    bool isConst;
};
struct MaxPoolTensorWrapper
{
    MaxPoolTensorWrapper(void* tensor_ptr, aclDataType real_type, aclFormat origin_format, aclFormat npu_format,
                        int64_t num_dims, const int64_t* dims, uint32_t tensor_size)
                    : tensor_ptr(tensor_ptr), real_type(real_type), origin_format(origin_format), npu_format(npu_format),
                        num_dims(num_dims), dims(dims), tensor_size(tensor_size) {}
    void * tensor_ptr;
    aclDataType real_type;
    aclFormat origin_format;
    aclFormat npu_format;
    int64_t num_dims;
    const int64_t* dims;
    uint32_t tensor_size;
};
class NpuCommand 
{
public:
    NpuCommand()
    {
        op_attr = aclopCreateAttr();
    }
    ~NpuCommand()
    {
        // do nothing, can not release resource, because of multi-thread or
        // queue-enable        
    }
    NpuCommand& OpName(const char* op_name);
    NpuCommand& Input(user_op::Tensor* input, std::string origin_format = "channels_nd", std::string desc_name = "",
                        std::string real_type = "");
    NpuCommand& Input(AclTensorWrapper& wrap);
    NpuCommand& Input(MaxPoolTensorWrapper& wrap);
    //NpuCommand& Input(std::vector<int32_t> &v);
    NpuCommand& Input();
    //NpuCommand& Input( NpuCommandSimpleTensorDesc input);
    NpuCommand& Output( user_op::Tensor* output, std::string format = "channels_nd", std::string desc_name = "",
                        std::string real_type = "");
    NpuCommand& Output(AclTensorWrapper& wrap);
    NpuCommand& Output(MaxPoolTensorWrapper& wrap);
    NpuCommand& Output();
    //NpuCommand& Output( NpuCommandSimpleTensorDesc output);
    // NpuCommand& InputDesc(const user_op::TensorDesc* input, std::string format);
    // NpuCommand& OutputDesc(const user_op::TensorDesc* output, std::string format);
    NpuCommand& Attr(std::string &&name, std::vector<int32_t> v);
    NpuCommand& Attr(std::string &&name, std::vector<int64_t> v);
    NpuCommand& Attr(std::string &&name, float f);
    NpuCommand& Attr(std::string &&name, bool b);
    NpuCommand& Attr(std::string &&name, int64_t i);
    NpuCommand& Attr(std::string &&name, std::string value);
    NpuCommand& Stream(aclrtStream stream);
    void Check();
    NpuCommand& Run();
private:
    struct CommandParam
    {
        // std::vector<user_op::Tensor*> inputs;
        // std::vector<user_op::Tensor*> outputs;
        std::vector<aclTensorDesc*> in_descs;
        std::vector<aclTensorDesc*> out_descs;
        std::vector<aclDataBuffer*> in_buffers;
        std::vector<aclDataBuffer*> out_buffers;
    };
    CommandParam command_param;
    aclopAttr *op_attr;
    aclrtStream stream;
    std::string op_name;

};  


}// namespace oneflow

#endif // WITH_NPU
#endif // ONEFLOW_USER_OPS_NPU_COMMAND_H_