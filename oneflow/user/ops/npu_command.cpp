#include <utility>
#include <map>
#include <algorithm>
#include "Python.h"
#include "oneflow/user/ops/npu_command.h"
#include "oneflow/core/device/npu_util.h"

namespace oneflow
{
aclDataType dataTypeMap(DataType type)
{
    if(datatype_map.find(type)!=datatype_map.end()) return datatype_map[type];
    return ACL_DT_UNDEFINED;
}

void PrintResult(void * out_buffers, uint32_t out_tensor_size, int data_len){
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
    if(data_len==2)
    {
        float16 *outdata = reinterpret_cast<float16*>(hostBuffer);

        for(int i=0;i<100;++i) 
        {
            std::cout<<outdata[i]<<" ";
        }
        std::cout<<std::endl;
    }
    else
    {
        float *outdata = reinterpret_cast<float*>(hostBuffer);

        for(int i=0;i<100;++i) 
        {
            std::cout<<outdata[i]<<" ";
        }
        std::cout<<std::endl;    
    }
}
void PrintResult(user_op::Tensor* out)
{
    PrintResult(out->mut_dptr<void>(),
                out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()),
                GetSizeOfDataType(out->data_type()));
}

aclTensorDesc* getTensorDesc(user_op::Tensor* tensor, std::string format)
{
    aclTensorDesc* descCast = aclCreateTensorDesc(dataTypeMap(tensor->data_type()), 
                                    tensor->shape().NumAxes(), 
                                    tensor->shape().ptr(), 
                                    format_map[format]);
    NOT_NULLPTR_CHECK(descCast);    
    return descCast;
}
aclTensorDesc* getTensorDesc(AclTensorWrapper& wrap)
{
    aclTensorDesc* descCast = aclCreateTensorDesc(wrap.data_type, 
                                    wrap.num_dims, 
                                    wrap.dims, 
                                    wrap.format);
    NOT_NULLPTR_CHECK(descCast);    
    return descCast;
}
// aclTensorDesc* getTensorDesc(std::vector<int32_t>& v)
// {
//     aclTensorDesc* descCast = aclCreateTensorDesc(ACL_INT32, 
//                                     1, 
//                                     (size_t*)&v.size(), 
//                                     wrap.format);
//     NOT_NULLPTR_CHECK(descCast);    
//     return descCast;
// }
aclDataBuffer* getDataBuffer(user_op::Tensor* tensor)
{
    aclDataBuffer* data_buffer = aclCreateDataBuffer(tensor->mut_dptr<void>(), 
                                                tensor->shape().elem_cnt() * GetSizeOfDataType(tensor->data_type())); 
    NOT_NULLPTR_CHECK(data_buffer);
    return data_buffer;
}
aclDataBuffer* getDataBuffer(AclTensorWrapper& wrap)
{
    aclDataBuffer* data_buffer = aclCreateDataBuffer(wrap.tensor_ptr, wrap.tensor_size); 
    NOT_NULLPTR_CHECK(data_buffer);
    return data_buffer;
}
// aclDataBuffer* getDataBuffer(std::vector<int32_t>& v)
// {
//     aclDataBuffer* data_buffer = aclCreateDataBuffer(v.data(), v.size()*4); 
//     NOT_NULLPTR_CHECK(data_buffer);
//     return data_buffer;
// }
void vector32To64(std::vector<int32_t>& src, std::vector<int64_t>& dst)
{
    for(int32_t i: src)
    {
        dst.push_back(i);
    }
}
NpuCommand& NpuCommand::OpName(const char* op_name)
{
    this->op_name = op_name;
    return *this;
}

NpuCommand& NpuCommand::Input(user_op::Tensor* input, std::string format)
{
    // generate DataBuffer
    command_param.in_buffers.push_back(getDataBuffer(input));
    // generate TensorDesc
    command_param.in_descs.push_back(getTensorDesc(input,format));
    return *this;
}
NpuCommand& NpuCommand::Input(AclTensorWrapper& wrap)
{
    // fix : use NpuAllocator to malloc
    OF_NPU_CHECK(aclrtMalloc(&wrap.tensor_ptr, wrap.tensor_size, ACL_MEM_MALLOC_NORMAL_ONLY));//dck_caution_here
    OF_NPU_CHECK(aclrtMemcpy(wrap.tensor_ptr,wrap.tensor_size,wrap.data_ptr,wrap.tensor_size,ACL_MEMCPY_HOST_TO_DEVICE));
    // generate DataBuffer
    command_param.in_buffers.push_back(getDataBuffer(wrap));
    // generate TensorDesc
    command_param.in_descs.push_back(getTensorDesc(wrap));
    return *this;
}
NpuCommand& NpuCommand::Input()
{
    command_param.in_descs.push_back(
                            aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED));
    return *this;
}
NpuCommand& NpuCommand::Output(user_op::Tensor* output, std::string format)
{
    // generate DataBuffer
    command_param.out_buffers.push_back(getDataBuffer(output));
    // generate TensorDesc
    command_param.out_descs.push_back(getTensorDesc(output,format));
    return *this;
}
NpuCommand& NpuCommand::Output(AclTensorWrapper& wrap)
{
    // fix : use NpuAllocator to malloc
    OF_NPU_CHECK(aclrtMalloc(&wrap.tensor_ptr, wrap.tensor_size, ACL_MEM_MALLOC_NORMAL_ONLY));//dck_caution_here
    // generate DataBuffer
    command_param.out_buffers.push_back(getDataBuffer(wrap));
    // generate TensorDesc
    command_param.out_descs.push_back(getTensorDesc(wrap));
    return *this;    
}
NpuCommand& NpuCommand::Output()
{
    command_param.out_descs.push_back(
                            aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED));
    return *this;
    
}
// NpuCommand& NpuCommand::InputDesc(const user_op::TensorDesc* input, std::string format)
// {
//     aclTensorDesc* inputDescCast = aclCreateTensorDesc(dataTypeMap(input->data_type()), 
//                                     input->shape().dim_vec().size(), 
//                                     input->shape().dim_vec().data(), 
//                                     format_map[format]);
//     NOT_NULLPTR_CHECK(inputDescCast);
//     command_param.in_descs.push_back(inputDescCast);
//     return *this;
// }
// NpuCommand& NpuCommand::OutputDesc(const  user_op::TensorDesc* output, std::string format)
// {
//     aclTensorDesc* outputDescCast = aclCreateTensorDesc(dataTypeMap(output->data_type()), 
//                                     output->shape().dim_vec().size(), 
//                                     output->shape().dim_vec().data(), 
//                                     format_map[format]);
//     NOT_NULLPTR_CHECK(outputDescCast);
//     command_param.out_descs.push_back(outputDescCast);
//     return *this;
// }
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
NpuCommand& NpuCommand::Attr(std::string &&name, int64_t i)
{   
    OF_NPU_CHECK(aclopSetAttrInt(op_attr, name.c_str(), i));
    return *this;
}
NpuCommand& NpuCommand::Attr(std::string &&name, std::string value)
{   
    if(name=="data_format")
    {
        if(attr_format_map.find(value)==attr_format_map.end())
        {
           std::cout<<"data_format error: data_format not found in map"<<std::endl;
        }
        value = attr_format_map[value];
    }
    OF_NPU_CHECK(aclopSetAttrString(op_attr, name.c_str(), value.c_str()));
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
    return ;
}
NpuCommand& NpuCommand::Run()
{

    std::cout<<"InDescSize1:"<<aclGetTensorDescSize(command_param.in_descs[0])<<std::endl;
    std::cout<<"InDescSize2:"<<aclGetTensorDescSize(command_param.in_descs[1])<<std::endl;
    std::cout<<"OutDescSize1:"<<aclGetTensorDescSize(command_param.out_descs[0])<<std::endl;
    if (PyGILState_Check()){
        Py_BEGIN_ALLOW_THREADS
        OF_NPU_CHECK(aclopCompile(op_name.c_str(), 
                                    command_param.in_descs.size(), 
                                    command_param.in_descs.data(), 
                                    command_param.out_descs.size(), 
                                    command_param.out_descs.data(), 
                                    op_attr,
                                    ACL_ENGINE_SYS, 
                                    ACL_COMPILE_SYS, 
                                    NULL));
        Py_END_ALLOW_THREADS
    }else{
        OF_NPU_CHECK(aclopCompile(op_name.c_str(), 
                                    command_param.in_descs.size(), 
                                    command_param.in_descs.data(), 
                                    command_param.out_descs.size(), 
                                    command_param.out_descs.data(), 
                                    op_attr, 
                                    ACL_ENGINE_SYS, 
                                    ACL_COMPILE_SYS, 
                                    NULL));
    }
    OF_NPU_CHECK(aclopExecuteV2(op_name.c_str(), 
                                command_param.in_descs.size(), 
                                command_param.in_descs.data(), 
                                command_param.in_buffers.data(), 
                                command_param.out_descs.size(), 
                                command_param.out_descs.data(),  
                                command_param.out_buffers.data(), 
                                op_attr,  
                                stream));
    return *this;
}
} // namespace