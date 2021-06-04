#include <jni.h>

#ifndef ONEFLOW_API_JAVA_EXAMPLE_H_
#define ONEFLOW_API_JAVA_EXAMPLE_H_

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jboolean JNICALL Java_com_example_App_isEnvInited(JNIEnv *, jobject);

JNIEXPORT void JNICALL Java_com_example_App_initEnv(JNIEnv *, jobject, jstring);

JNIEXPORT void JNICALL Java_com_example_App_helloWorld(JNIEnv *, jobject);

JNIEXPORT jint JNICALL Java_com_example_App_fibonacii(JNIEnv *, jobject, jint);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_EXAMPLE_H_
