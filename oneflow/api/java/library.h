#ifndef ONEFLOW_API_JAVA_LIBRARY_H_
#define ONEFLOW_API_JAVA_LIBRARY_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

// --------------- [global status area start] ---------------

// --------------- [global status area end] ---------------

JNIEXPORT void      JNICALL Java_org_oneflow_Library_initDefaultSession(JNIEnv* env, jobject obj);
JNIEXPORT jboolean  JNICALL Java_org_oneflow_Library_isEnvInited(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initEnv(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initScopeStack(JNIEnv* env, jobject obj, jstring jstr);
JNIEXPORT jboolean  JNICALL Java_org_oneflow_Library_isSessionInited(JNIEnv* env, jobject obj);
JNIEXPORT void      JNICALL Java_org_oneflow_Library_initSession(JNIEnv* env, jobject obj);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_LIBRARY_H_
