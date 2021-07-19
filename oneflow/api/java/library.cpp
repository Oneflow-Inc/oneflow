#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <future>

#include "jni.h"
#include "jni_md.h"
#include "oneflow/api/java/library.h"
#include "oneflow/api/python/framework/framework.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
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
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/api/java/session/session_api.h"
#include "oneflow/api/java/env/env_api.h"
#include "oneflow/api/java/job/job_api.h"


JNIEXPORT 
jint JNICALL Java_org_oneflow_InferenceSession_getEndian(JNIEnv* env, jobject obj) {
  return Endian();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setIsMultiClient(JNIEnv* env, jobject obj, jboolean is_multi_client) {
  return oneflow::SetIsMultiClient(is_multi_client).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initDefaultSession(JNIEnv* env, jobject obj) {
  return OpenDefaultSession();
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_InferenceSession_isEnvInited(JNIEnv* env, jobject obj) {
  return oneflow::IsEnvInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initEnv(JNIEnv* env, jobject obj, jstring env_proto_jstr) {
  std::string env_proto_str = ConvertToString(env, env_proto_jstr);

  return InitEnv(env_proto_str, false);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initScopeStack(JNIEnv* env, jobject obj) {
  return InitScopeStack();
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_InferenceSession_isSessionInited(JNIEnv* env, jobject obj) {
  return oneflow::IsSessionInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_initSession(JNIEnv* env, jobject obj, jstring config_proto) {
  std::string config_proto_str = ConvertToString(env, config_proto);
  return InitSession(config_proto_str);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_openJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_name) {
  std::string job_name_ = ConvertToString(env, job_name);

  return oneflow::JobBuildAndInferCtx_Open(job_name_).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setJobConfForCurJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_conf_proto) {
  std::string job_conf_proto_ = ConvertToString(env, job_conf_proto);

  return SetJobConfForCurJobBuildAndInferCtx(job_conf_proto_);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setScopeForCurJob(JNIEnv* env, jobject obj, jstring jstr) {
  // Todo: configuration
  std::string job_conf_proto = ConvertToString(env, jstr);

  return SetScopeForCurJob(job_conf_proto);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_curJobAddOp(JNIEnv* env, jobject obj, jstring op_conf_proto) {
  std::string op_conf_proto_ = ConvertToString(env, op_conf_proto);
  op_conf_proto_ = Subreplace(op_conf_proto_, "user_input", "input");
  op_conf_proto_ = Subreplace(op_conf_proto_, "user_output", "output");

  return CurJobAddOp(op_conf_proto_);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_completeCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  return oneflow::CurJobBuildAndInferCtx_Complete().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_rebuildCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  return oneflow::CurJobBuildAndInferCtx_Rebuild().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_unsetScopeForCurJob(JNIEnv* env, jobject obj) {
  return oneflow::ThreadLocalScopeStackPop().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_closeJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  return oneflow::JobBuildAndInferCtx_Close().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_startLazyGlobalSession(JNIEnv* env, jobject obj) {
  return oneflow::StartLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_loadCheckpoint(JNIEnv* env, jobject obj, jstring load_job, jobject path) {
  std::string load_job_name = ConvertToString(env, load_job);
  int64_t path_length = (*env).GetDirectBufferCapacity(path);
  void *path_address = (*env).GetDirectBufferAddress(path);

  return LoadCheckPoint(load_job_name, (signed char*) path_address, path_length);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_runSinglePushJob(JNIEnv* env, jobject obj, jobject data, jobject shape, jint dtype_code, jstring job_name, jstring op_name) {
  std::string job_name_ = ConvertToString(env, job_name);
  std::string op_name_ = ConvertToString(env, op_name);
  void *data_address = (*env).GetDirectBufferAddress(data);
  long *shape_address = (long*) (*env).GetDirectBufferAddress(shape);
  long shape_length = (*env).GetDirectBufferCapacity(shape);

  return RunPushJob(job_name_, op_name_, data_address, dtype_code, shape_address, shape_length);
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_runInferenceJob(JNIEnv* env, jobject obj, jstring jstr) {
  std::string inference_job_name = ConvertToString(env, jstr);

  return RunJob(inference_job_name);
}

JNIEXPORT
jobject JNICALL Java_org_oneflow_InferenceSession_runPullJob(JNIEnv* env, jobject obj, jstring job_name, jstring op_name) {  
  std::string job_name_ = ConvertToString(env, job_name);
  std::string op_name_ = ConvertToString(env, op_name);

  std::shared_ptr<PullTensor> pull_tensor = std::make_shared<PullTensor>();
  RunPullJobSync(job_name_, op_name_, pull_tensor);

  jbyteArray array = (*env).NewByteArray(pull_tensor->len_);
  (*env).SetByteArrayRegion(array, 0, pull_tensor->len_, reinterpret_cast<jbyte*>(pull_tensor->data_));
  jlongArray shapeArray = (*env).NewLongArray(pull_tensor->axes_);
  (*env).SetLongArrayRegion(shapeArray, 0, pull_tensor->axes_, pull_tensor->shape_);

  // call nativeNewTensor
  jclass tensorClass = (*env).FindClass("org/oneflow/Tensor");
  jmethodID mid = (*env).GetStaticMethodID(tensorClass, "nativeNewTensor", "([B[JI)Lorg/oneflow/Tensor;");
  jobject tensor = (*env).CallStaticObjectMethod(tensorClass, mid, array, shapeArray, pull_tensor->dtype_);

  return tensor;
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_stopLazyGlobalSession(JNIEnv* env, jobject obj) {
  return oneflow::StopLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_destroyLazyGlobalSession(JNIEnv* env, jobject obj) {
  return oneflow::DestroyLazyGlobalSession().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_destroyEnv(JNIEnv* env, jobject obj) {
  return oneflow::DestroyEnv().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_InferenceSession_setShuttingDown(JNIEnv* env, jobject obj) {
  return oneflow::SetShuttingDown();
}

JNIEXPORT
jstring JNICALL Java_org_oneflow_InferenceSession_getInterUserJobInfo(JNIEnv* env, jobject obj) {
  std::string inter_user_job_info = oneflow::GetSerializedInterUserJobInfo().GetOrThrow();
  return ConvertToJString(env, inter_user_job_info);
}
