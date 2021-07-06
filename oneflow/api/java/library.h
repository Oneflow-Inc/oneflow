#ifndef ONEFLOW_API_JAVA_LIBRARY_H_
#define ONEFLOW_API_JAVA_LIBRARY_H_

#include <jni.h>
#include "jni_md.h"

#ifdef __cplusplus
extern "C" {
#endif

// init
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_initDefaultSession(JNIEnv* env, jobject obj);
JNIEXPORT jboolean      JNICALL Java_org_oneflow_InferenceSession_isEnvInited(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_initEnv(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_initScopeStack(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT jboolean      JNICALL Java_org_oneflow_InferenceSession_isSessionInited(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_initSession(JNIEnv* env, jobject obj);

// compile
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_openJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_setJobConfForCurJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_setScopeForCurJob(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_curJobAddOp(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_completeCurJobBuildAndInferCtx(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_rebuildCurJobBuildAndInferCtx(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_unsetScopeForCurJob(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_closeJobBuildAndInferCtx(JNIEnv* env, jobject obj);

// launch
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_startLazyGlobalSession(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_loadCheckpoint(JNIEnv* env, jobject obj, jstring load_job, jbyteArray path);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_runSinglePushJob(JNIEnv* env, jobject obj, jbyteArray data, jlongArray shape, jint dtype_code, jstring job_name, jstring op_name);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_runInferenceJob(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT jobject       JNICALL Java_org_oneflow_InferenceSession_runPullJob(JNIEnv* env, jobject obj, jstring job_name, jstring op_name);

// clean
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_stopLazyGlobalSession(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_destroyLazyGlobalSession(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_destroyEnv(JNIEnv* env, jobject obj);
JNIEXPORT void          JNICALL Java_org_oneflow_InferenceSession_setShuttingDown(JNIEnv* env, jobject obj);

// others
JNIEXPORT jstring       JNICALL Java_org_oneflow_InferenceSession_getInterUserJobInfo(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_LIBRARY_H_
