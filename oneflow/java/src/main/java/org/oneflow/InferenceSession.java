package org.oneflow;

import com.google.protobuf.Descriptors;
import com.google.protobuf.InvalidProtocolBufferException;
import org.oneflow.core.common.Shape;
import org.oneflow.core.job.Env;
import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.job.InterUserJobInfoOuterClass.InterUserJobInfo;
import org.oneflow.core.job.JobConf;
import org.oneflow.core.job.JobConf.JobConfigProto;
import org.oneflow.core.job.JobSetOuterClass.ConfigProto;
import org.oneflow.core.job.ResourceOuterClass.Resource;
import org.oneflow.core.operator.OpConf.OperatorConf;
import org.oneflow.core.serving.SavedModelOuterClass.SavedModel;
import org.oneflow.core.serving.SavedModelOuterClass.GraphDef;
import org.oneflow.exception.CheckNullException;
import org.oneflow.exception.FileNotExistException;
import org.oneflow.exception.InitializationException;
import org.oneflow.util.ConfigConst;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;


public class InferenceSession {

    private final Option option;
    private String checkpointPath;
    private InterUserJobInfo interUserJobInfo;

    public InferenceSession() {
        this.option = new Option();
    }

    public InferenceSession(Option option) {
        this.option = option;
    }

    /**
     * Init the Env and Session
     */
    public void open() {
        OneFlow.setIsMultiClient(false);

        // 1, env init
        if (!OneFlow.isEnvInited()) {
            doEnvInit(option.getControlPort());

            // 2, scope init, Todo: pass this if CurrentMachineId not equal 0
            OneFlow.initScopeStack();
        }
        if (!OneFlow.isEnvInited()) {
            throw new InitializationException("Env is not inited correctly");
        }

        // 3, session init
        if (!OneFlow.isSessionInited()) {
            Resource.Builder resourceBuilder = Resource.newBuilder();
            resourceBuilder.setMachineNum(1); // Todo: machine num
            resourceBuilder.setEnableLegacyModelIo(true);
            if ("gpu".equals(option.getDeviceTag())) {
                resourceBuilder.setGpuDeviceNum(option.getDeviceNum());
                resourceBuilder.setCpuDeviceNum(0);
            }
            else {
                resourceBuilder.setGpuDeviceNum(0);
                resourceBuilder.setCpuDeviceNum(option.getDeviceNum());
            }

            ConfigProto.Builder builder = ConfigProto.newBuilder();
            builder.setSessionId(0); // Todo: session id
            builder.setResource(resourceBuilder.build());

            OneFlow.initSession(builder.build().toString());
        }
        if (!OneFlow.isSessionInited()) {
            throw new InitializationException("Session is not inited correctly");
        }
    }

    /**
     * try search the .pb/.prototxt file from given path and load it
     * Todo: support graph_name, signature_name, model version, model file basename
     * @param path
     */
    public void loadModel(String path) {
        String version = "1"; // Todo: support different version
        String savedModelPath = path + File.separator + version + File.separator;
        SavedModel model = readSavedModel(savedModelPath);
        this.checkpointPath= savedModelPath + File.separator + model.getCheckpointDir();

        // [Compile]
        String graphName = model.getDefaultGraphName();
        GraphDef graphDef = model.getGraphsOrThrow(graphName);

        // 1, prepare environment
        OneFlow.openJobBuildAndInferCtx(graphName);

        // set signature
        String signature_name = graphDef.getDefaultSignatureName();
        JobConf.JobSignatureDef jobSignatureDef = null;
        if (signature_name != null && !signature_name.equals("")) {
            jobSignatureDef = graphDef.getSignaturesOrThrow(signature_name);
        }

        JobConfigProto jobConfigProto = JobConfigProto.newBuilder()
                .setJobName(graphName)
                .setPredictConf(JobConf.PredictConf.newBuilder().build())
                .setSignature(jobSignatureDef)
                .build();

        OneFlow.setJobConfForCurJobBuildAndInferCtx(jobConfigProto.toString());

        // Todo: device_id_tags
        OneFlow.setScopeForCurJob(jobConfigProto.toString(), ConfigConst.DEVICE_IDS, option.getDeviceTag());

        // 2, do the compilation
        for (OperatorConf conf : graphDef.getOpListList()) {
            OneFlow.curJobAddOp(conf.toString());
        }
        OneFlow.completeCurJobBuildAndInferCtx();
        OneFlow.rebuildCurJobBuildAndInferCtx();

        // 3, clean the environment
        OneFlow.unsetScopeForCurJob();
        OneFlow.closeJobBuildAndInferCtx();
    }

    public void launch() {
        OneFlow.startLazyGlobalSession();

        String interUserJobInfo = OneFlow.getInterUserJobInfo();
        InterUserJobInfo info = null;
        try {
            info = InterUserJobInfo.parseFrom(interUserJobInfo.getBytes());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        if (info == null) {
            throw new CheckNullException("GetInterUserJobInfo failed");
        }
        this.interUserJobInfo = info;

        byte[] checkpointBytes = checkpointPath.getBytes();
        ByteBuffer checkpointBuffer = ByteBuffer.allocateDirect(checkpointBytes.length);
        checkpointBuffer.put(checkpointBytes);

        OneFlow.loadCheckpoint(info.getGlobalModelLoadJobName(), checkpointBuffer);
    }

    public Map<String, Tensor> run(String jobName, Map<String, Tensor> tensorMap) {
        // Push
        Map<String, String> inputNameToJobName = interUserJobInfo.getInputOrVarOpName2PushJobNameMap();
        for (Map.Entry<String, String> entry : inputNameToJobName.entrySet()) {
            Tensor tensor = tensorMap.get(entry.getKey());

            OneFlow.runSinglePushJob(tensor.getDataBuffer(), tensor.getShapeBuffer(),
                    tensor.getDataType().code, entry.getValue(), entry.getKey());
        }

        // Inference
        OneFlow.runInferenceJob(jobName);

        // Pull
        Map<String, Tensor> resultMap = new HashMap<>();
        for (Map.Entry<String, String> entry : interUserJobInfo.getOutputOrVarOpName2PullJobNameMap().entrySet()) {
            Tensor res = OneFlow.runPullJob(entry.getValue(), entry.getKey());
            resultMap.put(entry.getKey(), res);
        }

        return resultMap;
    }

    public void close() {
        OneFlow.stopLazyGlobalSession();
        OneFlow.destroyLazyGlobalSession();
        OneFlow.destroyEnv();
        OneFlow.setShuttingDown();
    }

    private static void doEnvInit(int port) {
        // reference: env_util.py 365 line
        EnvProto envProto = EnvProto.newBuilder()
                .addMachine(Env.Machine.newBuilder()
                        .setId(ConfigConst.MACHINE_ID)
                        .setAddr(ConfigConst.LOOPBACK))
                .setCtrlPort(port)
                .build(); // Todo: setId
        OneFlow.initEnv(envProto.toString());
    }

    private SavedModel readSavedModel(String path) {
        File file = null;
        for (String filename : ConfigConst.MODEL_FILENAMES) {
            File modelFile = new File(path + filename);
            if (modelFile.exists()) {
                file = modelFile;
                break;
            }
        }
        if (file == null) {
            throw new FileNotExistException(".pb/.proto file not exist");
        }

        SavedModel model = null;
        try (InputStream fis = new FileInputStream(file)) {
            model = SavedModel.parseFrom(fis);
        }
        catch (IOException e) {
            e.printStackTrace();
        }

        // Todo: need to handle if model is null
        return model;
    }
}
