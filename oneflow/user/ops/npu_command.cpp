#include <utility>
#include <map>
#include "Python.h"
#include "oneflow/user/ops/npu_command.h"
#include "oneflow/core/device/npu_util.h"

namespace oneflow
{

void PrintResult(void * out_buffers, uint32_t out_tensor_size){
    void* hostBuffer = nullptr;
    aclError ret = aclrtMallocHost(&hostBuffer, out_tensor_size);
    if (ret != ACL_ERROR_NONE) {
        std::cout<<"fail to print result, malloc host failed"<<std::endl;
	    
    }
    ret = aclrtMemcpy(hostBuffer, out_tensor_size, out_buffers,out_tensor_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        std::cout<<"fail to print result, memcpy device to host failed, errorCode is "<<ret<<std::endl;
        aclrtFreeHost(hostBuffer);
	    
    }

    float16 *outdata = reinterpret_cast<float16*>(hostBuffer);

    for(int i=0;i<100;++i) 
    {
        std::cout<<outdata[i]<<" ";
    }
    std::cout<<std::endl;
}

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
static std::map<std::string, aclFormat> format_map ={
                {"channel_last", ACL_FORMAT_NHWC},
                {"channel_first", ACL_FORMAT_NCHW},
                {"channel_nd", ACL_FORMAT_ND},
                {"channel_nc1hwc0",ACL_FORMAT_NC1HWC0},
            };
void vector32To64(std::vector<int32_t>& src, std::vector<int64_t>& dst)
{
    for(int32_t i: src)
    {
        dst.push_back(i);
    }
}
aclDataType dataTypeMap(DataType type)
{
    if(datatype_map.find(type)!=datatype_map.end()) return datatype_map[type];
    return ACL_DT_UNDEFINED;
}
NpuCommand& NpuCommand::OpName(const char* op_name)
{
    this->op_name = op_name;
    return *this;
}

NpuCommand& NpuCommand::Input(user_op::Tensor* input, std::string format)
{
    // generate DataBuffer
    command_param.inputs.push_back(input);
    aclDataBuffer* input_data = aclCreateDataBuffer(input->mut_dptr<void>(), 
                                                    input->shape().elem_cnt() * GetSizeOfDataType(input->data_type()));
    NOT_NULLPTR_CHECK(input_data);
    command_param.in_buffers.push_back(input_data);
    // generate TensorDesc
    aclTensorDesc* inputDescCast = aclCreateTensorDesc(dataTypeMap(input->data_type()), 
                                    input->shape().NumAxes(), 
                                    input->shape().ptr(), 
                                    format_map[format]);
    NOT_NULLPTR_CHECK(inputDescCast);
    command_param.acl_input_descs.push_back(inputDescCast);
    return *this;
}
NpuCommand& NpuCommand::Output(user_op::Tensor* output, std::string format)
{
    command_param.outputs.push_back(output);
    // generate DataBuffer
    aclDataBuffer* output_data = aclCreateDataBuffer(output->mut_dptr<void>(), 
                                                output->shape().elem_cnt() * GetSizeOfDataType(output->data_type())); 
    NOT_NULLPTR_CHECK(output_data);
    command_param.out_buffers.push_back(output_data);
    // generate TensorDesc
    aclTensorDesc* outputDescCast = aclCreateTensorDesc(dataTypeMap(output->data_type()), 
                                    output->shape().NumAxes(), 
                                    output->shape().ptr(), 
                                    format_map[format]);
    NOT_NULLPTR_CHECK(outputDescCast);
    command_param.acl_output_descs.push_back(outputDescCast);
    return *this;
}
NpuCommand& NpuCommand::InputDesc(const user_op::TensorDesc* input, std::string format)
{
    aclTensorDesc* inputDescCast = aclCreateTensorDesc(dataTypeMap(input->data_type()), 
                                    input->shape().dim_vec().size(), 
                                    input->shape().dim_vec().data(), 
                                    format_map[format]);
    NOT_NULLPTR_CHECK(inputDescCast);
    command_param.acl_input_descs.push_back(inputDescCast);
    return *this;
}
NpuCommand& NpuCommand::OutputDesc(const  user_op::TensorDesc* output, std::string format)
{
    aclTensorDesc* outputDescCast = aclCreateTensorDesc(dataTypeMap(output->data_type()), 
                                    output->shape().dim_vec().size(), 
                                    output->shape().dim_vec().data(), 
                                    format_map[format]);
    NOT_NULLPTR_CHECK(outputDescCast);
    command_param.acl_output_descs.push_back(outputDescCast);
    return *this;
}
NpuCommand& NpuCommand::Attr(std::string &&name, std::vector<int32_t> v)
{
    std::vector<int64_t> temp;
    vector32To64(v,temp);
    Attr(std::move(name), temp);
    return *this;
}
NpuCommand& NpuCommand::Attr(std::string &&name, std::vector<int64_t> v)
{
    OF_NPU_CHECK(aclopSetAttrListInt(op_attr, name.c_str(), v.size(), v.data()));
    return *this;
}
NpuCommand& NpuCommand::Attr(std::string &&name, float f)
{   
    OF_NPU_CHECK(aclopSetAttrFloat(op_attr, name.c_str(), f));
    return *this;
}
NpuCommand& NpuCommand::Attr(std::string &&name, bool b)
{   
    OF_NPU_CHECK(aclopSetAttrBool(op_attr, name.c_str(), b));
    return *this;
}
NpuCommand& NpuCommand::Stream(aclrtStream stream)
{
    this->stream = stream;
    return *this;
}

void NpuCommand::Check()
{
    NPU_COMMAND_CHECK(op_name!="");
    NPU_COMMAND_CHECK(!command_param.inputs.empty());
    NPU_COMMAND_CHECK(!command_param.outputs.empty());
    NPU_COMMAND_CHECK(checkDatatypeIsConsistent(command_param.inputs));
    NPU_COMMAND_CHECK(checkDatatypeIsConsistent(command_param.outputs));
    NPU_COMMAND_CHECK(command_param.acl_input_descs.size()==0||
                            command_param.inputs.size()==command_param.acl_input_descs.size());
    NPU_COMMAND_CHECK(command_param.acl_output_descs.size()==0||
                            command_param.outputs.size()==command_param.acl_output_descs.size());
    return ;
}
int NpuCommand::checkDatatypeIsConsistent(std::vector<user_op::Tensor*>& vec)
{
    DataType inputDataTypeCast = vec[0]->data_type();
    for(int i=1;i<vec.size();++i)
    {
        if(vec[i]->data_type()!=inputDataTypeCast)
        {
            return 0;
        }
    }      
    return 1;  
}

NpuCommand& NpuCommand::Run()
{

     
    if (PyGILState_Check()){
        Py_BEGIN_ALLOW_THREADS
        OF_NPU_CHECK(aclopCompile(op_name.c_str(), 
                                    command_param.acl_input_descs.size(), 
                                    command_param.acl_input_descs.data(), 
                                    command_param.acl_output_descs.size(), 
                                    command_param.acl_output_descs.data(), 
                                    op_attr,
                                    ACL_ENGINE_SYS, 
                                    ACL_COMPILE_SYS, 
                                    NULL));
        Py_END_ALLOW_THREADS
    }else{
        OF_NPU_CHECK(aclopCompile(op_name.c_str(), 
                                    command_param.acl_input_descs.size(), 
                                    command_param.acl_input_descs.data(), 
                                    command_param.acl_output_descs.size(), 
                                    command_param.acl_output_descs.data(), 
                                    op_attr, 
                                    ACL_ENGINE_SYS, 
                                    ACL_COMPILE_SYS, 
                                    NULL));
    }
    OF_NPU_CHECK(aclopExecuteV2(op_name.c_str(), 
                                command_param.inputs.size(), 
                                command_param.acl_input_descs.data(), 
                                command_param.in_buffers.data(), 
                                command_param.outputs.size(), 
                                command_param.acl_output_descs.data(),  
                                command_param.out_buffers.data(), 
                                op_attr,  
                                stream));
    return *this;
}
} // namespace