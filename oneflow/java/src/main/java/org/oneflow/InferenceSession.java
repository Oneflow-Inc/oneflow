package org.oneflow;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;


public class InferenceSession {

    private final Option option;

    public InferenceSession(Option option) {
        this.option = option;
    }

    public void open() {
        OneFlow.setIsMultiClient(false);

        if (!OneFlow.isEnvInited()) {
            OneFlow.initEnv(option.getControlPort());
            if (OneFlow.currentMachineId() == 0) {
                OneFlow.initScopeStack();
            }
        }

        if (!OneFlow.isSessionInited()) {
            OneFlow.initSession(option.getDeviceTag());
        }

        loadModel(option.getSavedModelDir());
        launch();
    }

    private void loadModel(String path) {
        // Todo: support different version
        String version = option.getModelVersion();

        // Todo: check existence
        String savedModelPath = path + File.separator + version + File.separator;
        option.setFullPathName(savedModelPath + File.separator + "saved_model.pb");

        OneFlow.loadModel(option);
    }

    private void launch() {
        OneFlow.startLazyGlobalSession();

        String path = option.getSavedModelDir() + File.separator +
                option.getModelVersion() + File.separator +
                option.getCheckpointDir();
        byte[] checkpointBytes = path.getBytes();
        ByteBuffer checkpointBuffer = ByteBuffer.allocateDirect(checkpointBytes.length);
        checkpointBuffer.put(checkpointBytes);

        OneFlow.loadCheckpoint(checkpointBuffer);
    }

    public Map<String, Tensor> run(String jobName, Map<String, Tensor> tensorMap) {
        // push job names: [job, op, job, op, ...]
        String[] pushJobNames = OneFlow.getPushJobNames().split(",");
        for (int i = 0; i < pushJobNames.length; i += 2) {
            Tensor tensor = tensorMap.get(pushJobNames[i]);
            OneFlow.runSinglePushJob(tensor.getDataBuffer(), tensor.getShapeBuffer(),
                    tensor.getDataType().code, pushJobNames[i + 1], pushJobNames[i]);
        }

        // Inference
        OneFlow.runInferenceJob(jobName);

        // Pull
        String[] pullJobNames = OneFlow.getPullJobNames().split(",");
        Map<String, Tensor> resultMap = new HashMap<>();
        for (int i = 0; i < pullJobNames.length; i += 2) {
            Tensor res = OneFlow.runPullJob(pullJobNames[i + 1], pullJobNames[i]);
            resultMap.put(pullJobNames[i], res);
        }

        return resultMap;
    }

    public void close() {
        OneFlow.stopLazyGlobalSession();
        OneFlow.destroyLazyGlobalSession();
        OneFlow.destroyEnv();
        OneFlow.setShuttingDown();
    }

}
