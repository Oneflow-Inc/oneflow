#include <utility>
#include <map>
#include <algorithm>
#include "Python.h"
#include "oneflow/user/ops/npu_command.h"
#include "oneflow/core/device/npu_util.h"

namespace oneflow
{
void testblock()
{
  int a = 1;
  return;
}

aclDataType dataTypeMap(DataType type)
{
    if(datatype_map.find(type)!=datatype_map.end()) return datatype_map[type];
    return ACL_DT_UNDEFINED;
}
#define PRINT_HOSTBUFFER(type)      type *outdata = reinterpret_cast<type*>(hostBuffer);\
        for(int i=0;i<100;++i) {std::cout<<outdata[i]<<" ";}

void PrintResult(void * out_buffers, uint32_t out_tensor_size, std::string type){
    void* hostBuffer = nullptr;
    aclError ret = aclrtMallocHost(&hostBuffer, out_tensor_size);
    if (ret != ACL_ERROR_NONE) {
        std::cout<<"fail to print result, malloc host failed"<<std::endl;
	    
    }
    std::cout<<"PrintResult "<<out_buffers<<std::endl;
    ret = aclrtMemcpy(hostBuffer, out_tensor_size, out_buffers,out_tensor_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        std::cout<<"fail to print result, memcpy device to host failed, errorCode is "<<ret<<std::endl;
        aclrtFreeHost(hostBuffer);
	    
    }
    if(type=="uint16")
    {
        PRINT_HOSTBUFFER(uint16_t)
    }
    else if(type=="uint64")
    {
        PRINT_HOSTBUFFER(uint64_t)
    }
    else if(type=="float")
    {
        PRINT_HOSTBUFFER(float)
    }
    std::cout<<std::endl;
}
void PrintResult(user_op::Tensor* out, std::string true_type)
{
    PrintResult(out->mut_dptr<void>(),
                out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()),
                true_type);
}
void PrintResult(void * out_buffers, uint32_t out_tensor_size, int data_len){
    void* hostBuffer = nullptr;
    aclError ret = aclrtMallocHost(&hostBuffer, out_tensor_size);
    if (ret != ACL_ERROR_NONE) {
        std::cout<<"fail to print result, malloc host failed"<<std::endl;
	    
    }
    std::cout<<"PrintResult "<<out_buffers<<std::endl;
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

aclTensorDesc* getTensorDesc(user_op::Tensor* tensor, std::string format, std::string real_type)
{
    aclDataType datatype = dataTypeMap(tensor->data_type());
    if(tensor->shape().NumAxes()==0)
    {
        if(real_type != "") datatype = datatype_string_map[real_type];
        aclTensorDesc* descCast = aclCreateTensorDesc(datatype, 
                                        0,
                                        nullptr,
                                        format_map[format]);
        NOT_NULLPTR_CHECK(descCast);    
        return descCast;
    }
    else
    {
        if(real_type != "") datatype = datatype_string_map[real_type];
        aclTensorDesc* descCast = aclCreateTensorDesc(datatype, 
                                        tensor->shape().NumAxes(), 
                                        tensor->shape().ptr(), 
                                        format_map[format]);
        NOT_NULLPTR_CHECK(descCast);    
        return descCast;       
    }
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
aclTensorDesc* getTensorDesc(MaxPoolTensorWrapper& wrap)
{
    aclTensorDesc* desc = aclCreateTensorDesc(wrap.real_type, 
                                    wrap.num_dims-1, 
                                    wrap.dims, 
                                    wrap.origin_format);
    NOT_NULLPTR_CHECK(desc);    
    aclSetTensorFormat(desc, wrap.npu_format);
    aclSetTensorShape(desc, wrap.num_dims, wrap.dims);
    return desc;
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
    if(tensor->shape().NumAxes()==0)
    {
        aclDataBuffer* data_buffer = aclCreateDataBuffer(tensor->mut_dptr<void>(), 
                                                    1 * GetSizeOfDataType(tensor->data_type())); 
        NOT_NULLPTR_CHECK(data_buffer);
        return data_buffer;
    }
    else
    {
        aclDataBuffer* data_buffer = aclCreateDataBuffer(tensor->mut_dptr<void>(), 
                                                    tensor->shape().elem_cnt() * GetSizeOfDataType(tensor->data_type())); 
        NOT_NULLPTR_CHECK(data_buffer);
        return data_buffer;
    }
}
aclDataBuffer* getDataBuffer(AclTensorWrapper& wrap)
{
    aclDataBuffer* data_buffer = aclCreateDataBuffer(wrap.tensor_ptr, wrap.tensor_size); 
    NOT_NULLPTR_CHECK(data_buffer);
    return data_buffer;
}
aclDataBuffer* getDataBuffer(MaxPoolTensorWrapper& wrap)
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

NpuCommand& NpuCommand::Input(user_op::Tensor* input, std::string format , std::string desc_name, std::string real_type)
{
    // generate DataBuffer
    command_param.in_buffers.push_back(getDataBuffer(input));
    // generate TensorDesc
    aclTensorDesc* desc = getTensorDesc(input, format, real_type);
    command_param.in_descs.push_back(desc);
    if(desc_name != "") aclSetTensorDescName(desc, desc_name.c_str());

    return *this;
}
NpuCommand& NpuCommand::Input(AclTensorWrapper& wrap)
{
    // fix : use NpuAllocator to malloc
    // generate DataBuffer
    command_param.in_buffers.push_back(getDataBuffer(wrap));
    // generate TensorDesc
    aclTensorDesc* desc = getTensorDesc(wrap);
    command_param.in_descs.push_back(desc);
    // setConsTensor
    if(wrap.isConst)
    {
        aclSetTensorConst(desc, wrap.data_ptr, wrap.tensor_size); 
    }
    return *this;
}
NpuCommand& NpuCommand::Input(MaxPoolTensorWrapper& wrap)
{
    // fix : use NpuAllocator to malloc
    // generate DataBuffer
    command_param.in_buffers.push_back(getDataBuffer(wrap));
    // generate TensorDesc
    aclTensorDesc* desc = getTensorDesc(wrap);
    command_param.in_descs.push_back(desc);
    // setConsTensor
    return *this;
}
NpuCommand& NpuCommand::Input()
{
    command_param.in_descs.push_back(
                            aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED));
    return *this;
}
NpuCommand& NpuCommand::Output(user_op::Tensor* output, std::string format, std::string desc_name, std::string real_type)
{
    // generate DataBuffer
    command_param.out_buffers.push_back(getDataBuffer(output));
    // generate TensorDesc
    aclTensorDesc* desc = getTensorDesc(output, format, real_type);
    command_param.out_descs.push_back(desc);
    if(desc_name != "") aclSetTensorDescName(desc, desc_name.c_str());
    return *this;
}
NpuCommand& NpuCommand::Output(AclTensorWrapper& wrap)
{
    // generate DataBuffer
    command_param.out_buffers.push_back(getDataBuffer(wrap));
    // generate TensorDesc
    command_param.out_descs.push_back(getTensorDesc(wrap));
    return *this;    
}
NpuCommand& NpuCommand::Output(MaxPoolTensorWrapper& wrap)
{
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

    // std::cout<<"InDescSize1:"<<aclGetTensorDescSize(command_param.in_descs[0])<<std::endl;
    // std::cout<<"InDescSize2:"<<aclGetTensorDescSize(command_param.in_descs[1])<<std::endl;
    // std::cout<<"OutDescSize1:"<<aclGetTensorDescSize(command_param.out_descs[0])<<std::endl;
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