#include <jni.h>
#include "oneflow/api/java/tensor.h"

JNIEXPORT
jlong JNICALL Java_org_oneflow_Tensor_newTensor(JNIEnv* env, jobject obj, jbyteArray data, jlongArray shape, jint d_type) {
    return 0;
}

JNIEXPORT
void JNICALL Java_org_oneflow_Tensor_free(JNIEnv* env, jobject obj, jlong tensor_ptr) {
    return;
}

JNIEXPORT
jbyteArray JNICALL Java_org_oneflow_Tensor_getData(JNIEnv* env, jobject obj, jlong tensor_ptr) {
    return NULL;
}
