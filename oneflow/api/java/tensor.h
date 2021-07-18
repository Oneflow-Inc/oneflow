#ifndef ONEFLOW_API_JAVA_TENSOR_H_
#define ONEFLOW_API_JAVA_TENSOR_H_

#include <jni.h>
#include "jni_md.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong         JNICALL Java_org_oneflow_Tensor_newTensor(JNIEnv* env, jobject obj, jbyteArray data, jlongArray shape, jint d_type);
JNIEXPORT void          JNICALL Java_org_oneflow_Tensor_free(JNIEnv* env, jobject obj, jlong tensor_ptr);
JNIEXPORT jbyteArray    JNICALL Java_org_oneflow_Tensor_getData(JNIEnv* env, jobject obj, jlong tensor_ptr);

#ifdef __cplusplus
}
#endif
#endif  // ONEFLOW_API_JAVA_TENSOR_H_
