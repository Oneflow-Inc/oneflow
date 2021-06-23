#ifndef ONEFLOW_API_JAVA_LIBRARY_H_
#define ONEFLOW_API_JAVA_LIBRARY_H_

#include <jni.h>
#include "jni_md.h"

#ifdef __cplusplus
extern "C" {
#endif

// init
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initDefaultSession(JNIEnv* env, jobject obj);
JNIEXPORT jboolean  JNICALL Java_org_oneflow_Library_isEnvInited(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initEnv(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initScopeStack(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT jboolean  JNICALL Java_org_oneflow_Library_isSessionInited(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initSession(JNIEnv* env, jobject obj);

// compile
JNIEXPORT void      JNICALL Java_org_oneflow_Library_openJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_setJobConfForCurJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_setScopeForCurJob(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_curJobAddOp(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_completeCurJobBuildAndInferCtx(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_rebuildCurJobBuildAndInferCtx(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_unsetScopeForCurJob(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_closeJobBuildAndInferCtx(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_LIBRARY_H_
