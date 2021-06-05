#ifndef ONEFLOW_API_JAVA_JOB_BUILD_JOB_BUILD_AND_INFER_H_
#define ONEFLOW_API_JAVA_JOB_BUILD_JOB_BUILD_AND_INFER_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_org_oneflow_job_JobBuildAndInferCtx_open(JNIEnv* env, jobject obj,
                                                                     jstring jstr);

JNIEXPORT jstring JNICALL Java_org_oneflow_job_JobBuildAndInferCtx_getCurrentJobName(JNIEnv* env,
                                                                                     jobject obj,
                                                                                     jstring jstr);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_JOB_BUILD_JOB_BUILD_AND_INFER_H_