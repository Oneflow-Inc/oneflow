#ifndef ONEFLOW_API_JAVA_ENV_ENV_H_
#define ONEFLOW_API_JAVA_ENV_ENV_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jboolean JNICALL Java_org_oneflow_env_Env_isEnvInited(JNIEnv* env, jobject obj);

JNIEXPORT void JNICALL Java_org_oneflow_env_Env_initEnv(JNIEnv* env, jobject obj, jstring jstr);
    
#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_ENV_ENV_H_
