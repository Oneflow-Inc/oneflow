#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <istream>
#include <memory>
#include <string>
#include <vector>
#include <future>

#include "jni.h"
#include "jni_md.h"
#include "oneflow/api/java/library.h"
#include "oneflow/api/python/framework/framework.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/job_build/job_build_and_infer_api.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session_api.h"
#include "oneflow/core/common/cfg.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/api/java/util/jni_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/shut_down_util.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/serving/saved_model.pb.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/api/java/session/session_api.h"
#include "oneflow/api/java/env/env_api.h"
#include "oneflow/api/java/job/job_api.h"


JNIEXPORT 
jint JNICALL Java_org_oneflow_OneFlow_getEndian(JNIEnv* env, jobject obj) {
  return Endian();
}

JNIEXPORT
jint JNICALL Java_org_oneflow_OneFlow_getNodeSize(JNIEnv* env, jobject obj) {
  return oneflow::GetNodeSize().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_setIsMultiClient(JNIEnv* env, jobject obj, jboolean is_multi_client) {
  return oneflow::SetIsMultiClient(is_multi_client).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_initDefaultSession(JNIEnv* env, jobject obj) {
  return OpenDefaultSession();
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_OneFlow_isEnvInited(JNIEnv* env, jobject obj) {
  return oneflow::IsEnvInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_initEnv(JNIEnv* env, jobject obj, jint ctrl_port) {
  return InitEnv(ctrl_port);
}

JNIEXPORT
jlong JNICALL Java_org_oneflow_OneFlow_currentMachineId(JNIEnv* env, jobject obj) {
  return CurrentMachineId();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_initScopeStack(JNIEnv* env, jobject obj) {
  return InitScopeStack();
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_OneFlow_isSessionInited(JNIEnv* env, jobject obj) {
  return oneflow::IsSessionInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_initSession(JNIEnv* env, jobject obj, jstring device_tag) {
  std::string device_tag_ = ConvertToString(env, device_tag);
  return InitSession(device_tag_);
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_loadModel(JNIEnv* env, jobject obj, jobject option) {
  jstring full_path_name = (jstring) GetOptionField(env, option, "fullPathName", "Ljava/lang/String;");
  std::string full_path_name_ = ConvertToString(env, full_path_name);
  
  oneflow::SavedModel saved_model;
  std::ifstream file;
  file.open(full_path_name_, std::ios::in);
  std::istream* file_istream = &file;
  saved_model.ParseFromIstream(file_istream);
  file.close();
  std::string checkpoint_dir = saved_model.checkpoint_dir();

  jstring checkpoint_dir_ = ConvertToJString(env, checkpoint_dir);
  SetStringField(env, option, "checkpointDir", "Ljava/lang/String;", checkpoint_dir_);

  // compile
  std::string graph_name = saved_model.default_graph_name();
  oneflow::GraphDef graph_def = saved_model.graphs().at(graph_name);

  oneflow::JobBuildAndInferCtx_Open(graph_name);
  
  oneflow::JobConfigProto job_config_proto;
  job_config_proto.set_job_name(graph_name);
  job_config_proto.mutable_predict_conf();
  // set signature
  if (graph_def.has_default_signature_name()) {
    auto signature_name = graph_def.default_signature_name();
    std::cout << signature_name << std::endl;
    auto job_signature_def = graph_def.signatures().at(signature_name);
    job_config_proto.mutable_signature()->CopyFrom(job_signature_def);
  }
  std::cout << job_config_proto.DebugString() << std::endl;
  SetJobConfForCurJobBuildAndInferCtx(job_config_proto.DebugString());
  SetScopeForCurJob(job_config_proto.DebugString(), "0:0", "gpu");

  auto op_list = graph_def.op_list();
  for (auto op_conf = op_list.begin(); op_conf != op_list.end(); op_conf++) {
    CurJobAddOp(op_conf->DebugString());
  }
  oneflow::CurJobBuildAndInferCtx_Complete();
  oneflow::CurJobBuildAndInferCtx_Rebuild();
  oneflow::ThreadLocalScopeStackPop();
  oneflow::JobBuildAndInferCtx_Close();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_openJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_name) {
  std::string job_name_ = ConvertToString(env, job_name);

  return oneflow::JobBuildAndInferCtx_Open(job_name_).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_setJobConfForCurJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_conf_proto) {
  std::string job_conf_proto_ = ConvertToString(env, job_conf_proto);

  return SetJobConfForCurJobBuildAndInferCtx(job_conf_proto_);
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_setScopeForCurJob(JNIEnv* env, jobject obj, jstring job_conf_proto, jstring ids, jstring device) {
  std::string job_conf_proto_ = ConvertToString(env, job_conf_proto);
  std::string ids_ = ConvertToString(env, ids);
  std::string device_ = ConvertToString(env, device);

  return SetScopeForCurJob(job_conf_proto_, ids_, device_);
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_curJobAddOp(JNIEnv* env, jobject obj, jstring op_conf_proto) {
  std::string op_conf_proto_ = ConvertToString(env, op_conf_proto);
  return CurJobAddOp(op_conf_proto_);
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_completeCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  return oneflow::CurJobBuildAndInferCtx_Complete().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_rebuildCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  return oneflow::CurJobBuildAndInferCtx_Rebuild().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_unsetScopeForCurJob(JNIEnv* env, jobject obj) {
  return oneflow::ThreadLocalScopeStackPop().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_closeJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  return oneflow::JobBuildAndInferCtx_Close().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_startLazyGlobalSession(JNIEnv* env, jobject obj) {
  return oneflow::StartLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_loadCheckpoint(JNIEnv* env, jobject obj, jobject path) {
  std::string inter_user_job_info = oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
  oneflow::InterUserJobInfo info;
  info.ParseFromString(inter_user_job_info);
  std::string load_job_name = info.global_model_load_job_name();

  int64_t path_length = (*env).GetDirectBufferCapacity(path);
  void *path_address = (*env).GetDirectBufferAddress(path);

  return LoadCheckPoint(load_job_name, (signed char*) path_address, path_length);
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_runSinglePushJob(JNIEnv* env, jobject obj, jobject data, jobject shape, jint dtype_code, jstring job_name, jstring op_name) {
  std::string job_name_ = ConvertToString(env, job_name);
  std::string op_name_ = ConvertToString(env, op_name);
  void *data_address = (*env).GetDirectBufferAddress(data);
  long *shape_address = (long*) (*env).GetDirectBufferAddress(shape);
  long shape_length = (*env).GetDirectBufferCapacity(shape);

  return RunPushJob(job_name_, op_name_, data_address, dtype_code, shape_address, shape_length);
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_runInferenceJob(JNIEnv* env, jobject obj, jstring jstr) {
  std::string inference_job_name = ConvertToString(env, jstr);

  return RunJob(inference_job_name);
}

JNIEXPORT
jobject JNICALL Java_org_oneflow_OneFlow_runPullJob(JNIEnv* env, jobject obj, jstring job_name, jstring op_name) {  
  std::string job_name_ = ConvertToString(env, job_name);
  std::string op_name_ = ConvertToString(env, op_name);

  std::shared_ptr<PullTensor> pull_tensor = std::make_shared<PullTensor>();
  RunPullJobSync(job_name_, op_name_, pull_tensor);

  jbyteArray array = (*env).NewByteArray(pull_tensor->len_);
  (*env).SetByteArrayRegion(array, 0, pull_tensor->len_, reinterpret_cast<jbyte*>(pull_tensor->data_));
  jlongArray shapeArray = (*env).NewLongArray(pull_tensor->axes_);
  (*env).SetLongArrayRegion(shapeArray, 0, pull_tensor->axes_, pull_tensor->shape_);

  // call nativeNewTensor
  // Todo: Exception handle
  jclass tensor_class = (*env).FindClass("org/oneflow/Tensor");
  jmethodID mid = (*env).GetStaticMethodID(tensor_class, "nativeNewTensor", "([B[JI)Lorg/oneflow/Tensor;");
  jobject tensor = (*env).CallStaticObjectMethod(tensor_class, mid, array, shapeArray, pull_tensor->dtype_);

  return tensor;
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_stopLazyGlobalSession(JNIEnv* env, jobject obj) {
  return oneflow::StopLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_destroyLazyGlobalSession(JNIEnv* env, jobject obj) {
  return oneflow::DestroyLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_destroyEnv(JNIEnv* env, jobject obj) {
  return oneflow::DestroyEnv().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_OneFlow_setShuttingDown(JNIEnv* env, jobject obj) {
  return oneflow::SetShuttingDown();
}

JNIEXPORT
jstring JNICALL Java_org_oneflow_OneFlow_getInterUserJobInfo(JNIEnv* env, jobject obj) {
  std::string inter_user_job_info = oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
  return ConvertToJString(env, inter_user_job_info);
}

JNIEXPORT
jstring JNICALL Java_org_oneflow_OneFlow_getPushJobNames(JNIEnv* env, jobject obj) {
  std::string inter_user_job_info = oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
  oneflow::InterUserJobInfo info;
  info.ParseFromString(inter_user_job_info);

  std::string push_job_names;
  auto input2push = info.input_or_var_op_name2push_job_name();
  for (auto iter = input2push.begin(); iter != input2push.end(); iter++) {
    push_job_names = push_job_names + iter->first + "," + iter->second + ",";;
  }
  
  jstring res = ConvertToJString(env, push_job_names);
  return res;
}

JNIEXPORT
jstring JNICALL Java_org_oneflow_OneFlow_getPullJobNames(JNIEnv* env, jobject obj) {
  std::string inter_user_job_info = oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
  oneflow::InterUserJobInfo info;
  info.ParseFromString(inter_user_job_info);

  std::string pull_job_names;
  auto output2pull = info.output_or_var_op_name2pull_job_name();
  for (auto iter = output2pull.begin(); iter != output2pull.end(); iter++) {
    pull_job_names = pull_job_names + iter->first + "," + iter->second + ",";
  }

  jstring res = ConvertToJString(env, pull_job_names);
  return res;
}
