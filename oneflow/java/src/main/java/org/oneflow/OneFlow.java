package org.oneflow;

import java.nio.Buffer;
import java.nio.ByteOrder;

class OneFlow {
    static {
        System.loadLibrary("oneflow_java");

        // Default Initialization: beyond import oneflow as flow
        OneFlow.initDefaultSession();
        if (getEndian() == 0) {
            Tensor.endian = ByteOrder.BIG_ENDIAN;
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("shutting down OneFlow");
            OneFlow.destroyEnv();
            OneFlow.setShuttingDown();
        }));
    }

    // 0 for big endian, 1 for little endian
    static native int getEndian();
    static native int getNodeSize();

    // init
    static native void setIsMultiClient(boolean isMultiClient);
    static native void initDefaultSession();
    static native boolean isEnvInited();
    static native void initEnv(int port);
    static native long currentMachineId();
    static native void initScopeStack();
    static native boolean isSessionInited();
    static native void initSession(String deviceTag);

    // compile
    static native void loadModel(Option option);
    static native void openJobBuildAndInferCtx(String jobName);
    static native void setJobConfForCurJobBuildAndInferCtx(String jobConfProto);
    static native void setScopeForCurJob(String jobConfProto, String ids, String device);
    static native void curJobAddOp(String opConfProto);
    static native void completeCurJobBuildAndInferCtx();
    static native void rebuildCurJobBuildAndInferCtx();
    static native void unsetScopeForCurJob();
    static native void closeJobBuildAndInferCtx();

    // launch
    static native void startLazyGlobalSession();
    static native void loadCheckpoint(Buffer path);

    // forward
    static native String getPushJobNames();
    static native String getPullJobNames();
    static native void runSinglePushJob(Buffer data,
                                        Buffer shape,
                                        int dTypeCode,
                                        String jobName,
                                        String opName);
    static native void runInferenceJob(String jobName);
    static native Tensor runPullJob(String jobName, String opName);

    // clean
    static native void stopLazyGlobalSession();
    static native void destroyLazyGlobalSession();
    static native void destroyEnv();
    static native void setShuttingDown();

    // others
    static native String getInterUserJobInfo();
}
