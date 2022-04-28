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
#define NOT_NULLPTR_CHECK(x) if(x==nullptr) {                                           \
          std::cout<<"Get nullptr error, create input "<<#x<<" failed"<<std::endl; \
        }
void PrintResult(void * out_buffers, uint32_t out_tensor_size);
void vector32To64(std::vector<int32_t>& src, std::vector<int64_t>& dst);

#define NPU_COMMAND_CHECK(x) if(!(x)) {\
    std::cout<<"npu_command check fail "<<#x<<std::endl;\
    return ;\
    }
template<typename dtype>
struct NpuCommandSimpleTensorDesc
{
    dtype* data_ptr;
    size_t count;
    std::string format;
    DataType data_type = kInvalid;
    NpuCommandSimpleTensorDesc(dtype* ptr, size_t cnt, std::string fmt)
                                :data_ptr(ptr), count(cnt), format(fmt) {}
};
template<>
struct NpuCommandSimpleTensorDesc<float>
{
    float* data_ptr;
    size_t count;
    std::string format;
    DataType data_type = kFloat;
    NpuCommandSimpleTensorDesc(dtype* ptr, size_t cnt, std::string fmt)
                                :data_ptr(ptr), count(cnt), format(fmt) {}
};
template<>
struct NpuCommandSimpleTensorDesc<float16>
{
    float16* data_ptr;
    size_t count;
    std::string format;
    DataType data_type = kFloat16;
    NpuCommandSimpleTensorDesc(dtype* ptr, size_t cnt, std::string fmt)
                                :data_ptr(ptr), count(cnt), format(fmt) {}
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
    NpuCommand& Input( user_op::Tensor* input, std::string format);
    //NpuCommand& Input( NpuCommandSimpleTensorDesc input);
    NpuCommand& Output( user_op::Tensor* output, std::string format);
    //NpuCommand& Output( NpuCommandSimpleTensorDesc output);
    NpuCommand& InputDesc(const user_op::TensorDesc* input, std::string format);
    NpuCommand& OutputDesc(const user_op::TensorDesc* output, std::string format);
    NpuCommand& Attr(std::string &&name, std::vector<int32_t> v);
    NpuCommand& Attr(std::string &&name, std::vector<int64_t> v);
    NpuCommand& Attr(std::string &&name, float f);
    NpuCommand& Attr(std::string &&name, bool b);
    NpuCommand& Stream(aclrtStream stream);
    void Check();
    int checkDatatypeIsConsistent(std::vector<user_op::Tensor*>& vec);
    int checkDatatype(user_op::Tensor* tensor);
    NpuCommand& Run();
private:
    struct CommandParam
    {
        std::vector<user_op::Tensor*> inputs;
        std::vector<user_op::Tensor*> outputs;
        std::vector<aclTensorDesc*> acl_input_descs;
        std::vector<aclTensorDesc*> acl_output_descs;
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