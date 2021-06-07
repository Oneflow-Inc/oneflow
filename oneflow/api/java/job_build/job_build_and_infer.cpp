#include "oneflow/api/java/job_build/job_build_and_infer.h"
#include <jni.h>
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/java/util/string_util.h"

JNIEXPORT
void JNICALL Java_org_oneflow_job_JobBuildAndInferCtx_open(JNIEnv* env, jobject obj,
                                                           jstring job_name) {
  return oneflow::JobBuildAndInferCtx_Open(convert_jstring_to_string(env, job_name)).GetOrThrow();
}

JNIEXPORT
jstring JNICALL Java_org_oneflow_job_JobBuildAndInferCtx_getCurrentJobName(JNIEnv* env, jobject obj,
                                                                           jstring jstr) {
  std::string current_job_name = oneflow::JobBuildAndInferCtx_GetCurrentJobName().GetOrThrow();
  return convert_string_to_jstring(env, current_job_name);
}
