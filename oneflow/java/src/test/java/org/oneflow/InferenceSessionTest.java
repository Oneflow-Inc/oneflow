package org.oneflow;

import org.junit.BeforeClass;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.math.BigInteger;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import static org.junit.Assert.*;

public class InferenceSessionTest {

    private final static int BUFFER_SIZE = 1024;

    private final static String ZIP_FILE = "mnist_test.zip";
    private final static String ZIP_MD5 = "cf18aafc32e923d1289ed0ccfae96f7e";
    private final static String ZIP_URL =
            "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/java-serving/mnist_test.zip";
    private final static String UNZIP_DIR = "mnist_test";

    private final static String[] IMAGE_FILES = {
            "00000000_7.png", "00000002_1.png", "00000004_4.png", "00000006_4.png", "00000008_5.png",
            "00000001_2.png", "00000003_0.png", "00000005_1.png", "00000007_9.png", "00000009_9.png"
    };
    private final static float[][] IMAGE_MODEL1_LOGITS = {
            {-129.57167f, -89.084816f, -139.21355f, -103.455025f, -9.179366f,
                    -69.568474f, -133.39594f, -16.204329f, -114.90876f, -47.933548f},
            {-157.07368f, -48.71157f, -154.10773f, -155.76338f, -16.631721f,
                    -48.054104f, -94.90756f, -99.00399f, -129.38669f, -77.54886f},
            {-131.89546f, -99.77963f, -110.64235f, -135.06781f, 54.82699f,
                    -72.111565f, -90.52143f, -53.132538f, -121.65344f, -55.689552f},
            {-149.22449f, -90.75103f, -145.55164f, -134.30214f, 50.0643f,
                    -61.743526f, -99.326935f, -53.22419f, -114.44637f, -53.091938f},
            {-124.82982f, -72.2742f, -105.80667f, -132.53188f, 24.909275f,
                    -15.583976f, -41.318104f, -96.46596f, -87.08713f, -74.991135f},
            {-77.40619f, -32.485638f, -44.314667f, -111.69051f, -30.799593f,
                    -45.85844f, -27.795767f, -97.81223f, -81.61974f, -86.59527f},
            {-58.406326f, -86.02751f, -96.20607f, -118.30356f, 0.8787302f,
                    -57.716908f, -49.96166f, -56.63072f, -68.43286f, -62.825005f},
            {-156.19606f, -44.278812f, -163.46623f, -152.99944f, -17.676971f,
                    -50.860874f, -103.65146f, -92.46598f, -122.09205f, -71.87215f},
            {-162.38156f, -60.381863f, -122.38432f, -140.63043f, 39.856358f,
                    -68.038284f, -121.28112f, -71.01519f, -151.50551f, -27.652596f},
            {-139.47165f, -115.41351f, -161.98169f, -135.65639f, 34.886948f,
                    -69.54524f, -118.65837f, -51.901855f, -105.19491f, -27.991352f}
    };
    private final static float[][] IMAGE_MODEL2_LOGITS = {
            {-130.93362f, -72.18874f, -165.4578f, -114.83206f, -21.068695f,
                    -88.18619f, -151.45547f, 19.964624f, -142.91219f, -43.09779f},
            {-166.03882f, -1.3469127f, -171.76181f, -175.45944f, -20.448563f,
                    -79.79947f, -132.58748f, -60.517467f, -165.38629f, -68.77712f},
            {-116.715294f, -70.46023f, -123.3226f, -153.3579f, 44.70533f,
                    -88.71307f, -95.11289f, -10.951504f, -131.17422f, -57.423725f},
            {-151.22302f, -61.256893f, -161.53299f, -145.1139f, 49.09799f,
                    -72.483444f, -121.79374f, -20.381575f, -129.54608f, -48.442257f},
            {-103.363495f, -55.77169f, -102.5316f, -153.61868f, 20.056509f,
                    -34.20384f, -56.19096f, -69.902916f, -92.93077f, -71.56784f},
            {-90.955536f, -8.2702265f, -53.13394f, -141.75288f, -39.389755f,
                    -60.695724f, -43.814228f, -74.59845f, -105.5987f, -83.68641f},
            {-34.08243f, -73.38965f, -103.65884f, -124.556f, -0.71969944f,
                    -60.068783f, -51.103535f, -18.294123f, -85.12092f, -58.302242f},
            {-162.33029f, 8.184846f, -181.80286f, -166.96364f, -21.640394f,
                    -83.26369f, -139.6628f, -51.001266f, -156.80318f, -65.17207f},
            {-152.81104f, -29.032887f, -148.63463f, -155.9041f, 38.25778f,
                    -91.30199f, -146.35764f, -31.38757f, -161.32152f, -22.303255f},
            {-120.75517f, -66.02598f, -179.05809f, -137.83466f, 26.579493f,
                    -86.210655f, -135.6882f, -8.564451f, -118.15144f, -21.3309f}
    };
    private final static float[] IMAGE_MODEL3_LOGITS = {
            -126.393394f, -105.65245f, -174.31293f, -103.66381f, -29.701872f,
            -51.540207f, -143.43855f, -12.14451f, -112.93034f, -40.30441f,
            -146.96033f, -56.29207f, -179.6213f, -153.55287f, -45.178135f,
            -45.364582f, -120.16257f, -100.640816f, -143.6211f, -68.58198f,
            -115.41144f, -96.740746f, -133.62782f, -141.78201f, 21.6788f,
            -56.994793f, -97.83147f, -44.019424f, -121.500786f, -45.909588f,
            -138.82664f, -102.61933f, -176.05437f, -132.97458f, 35.305008f,
            -35.82211f, -120.114075f, -54.043243f, -116.10012f, -37.80337f,
            -96.44919f, -84.8902f, -131.30237f, -142.67319f, -8.628632f,
            -9.1317425f, -55.398415f, -82.68884f, -83.33495f, -66.941475f,
            -66.87236f, -39.42447f, -61.490143f, -107.93371f, -56.34854f,
            -38.44101f, -44.598373f, -106.30037f, -93.28503f, -75.650185f,
            -31.168306f, -95.47076f, -122.687386f, -117.45813f, -13.716896f,
            -51.557846f, -59.256172f, -35.329865f, -74.02576f, -48.717777f,
            -148.68535f, -51.58836f, -190.09341f, -151.80173f, -44.12176f,
            -51.87872f, -126.28073f, -88.04969f, -134.81372f, -65.84527f,
            -153.44884f, -62.901814f, -163.33693f, -136.81006f, 23.749039f,
            -59.479866f, -123.63465f, -69.4092f, -147.9043f, -10.745205f,
            -130.49567f, -113.29153f, -181.8259f, -131.45947f, 20.267225f,
            -53.104675f, -125.89969f, -43.8397f, -94.868195f, -3.8976247f};
    private final static float DELTA = 0.0001f;

    /**
     * md5: https://stackoverflow.com/questions/5297552/calculate-md5-hash-of-a-zip-file-in-java-program
     * unzip: https://www.baeldung.com/java-compress-and-uncompress
     * delete: https://www.baeldung.com/java-delete-directory
     */
    @BeforeClass
    public static void downloadResources() throws NoSuchAlgorithmException, IOException {
        boolean needDownload = true;

        System.out.println("Checking file existence and calculating md5");
        File testZip = new File(ZIP_FILE);
        if (testZip.exists()) {
            MessageDigest digest = MessageDigest.getInstance("MD5");

            try (InputStream is = new FileInputStream(testZip)) {
                byte[] buffer = new byte[BUFFER_SIZE];
                int read;
                while((read = is.read(buffer)) > 0) {
                    digest.update(buffer, 0, read);
                }

                byte[] md5sum = digest.digest();
                BigInteger bigInt = new BigInteger(1, md5sum);
                String output = bigInt.toString(16);
                if(ZIP_MD5.equals(output)) {
                    needDownload = false;
                }
            }
            catch(IOException e) {
                throw new RuntimeException("Unable to process file for MD5", e);
            }
        }

        if (needDownload) {
            System.out.println("downloading resources");
            new FileOutputStream(ZIP_FILE).getChannel().transferFrom(
                    Channels.newChannel(new URL(ZIP_URL).openStream()), 0, Long.MAX_VALUE);
            testZip = new File(ZIP_FILE);
        }

        System.out.println("unzip resources");
        File unzipDir = new File(UNZIP_DIR);
        if (unzipDir.exists()) {
            Files.walk(unzipDir.toPath())
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }

        byte[] buffer = new byte[BUFFER_SIZE];
        ZipInputStream zis = new ZipInputStream(new FileInputStream(testZip));
        ZipEntry zipEntry = zis.getNextEntry();
        while (zipEntry != null) {
            File newFile = new File(String.valueOf(zipEntry));
            if (zipEntry.isDirectory()) {
                if (!newFile.isDirectory() && !newFile.mkdirs()) {
                    throw new IOException("Failed to create directory " + newFile);
                }
            }
            else {
                // fix for Windows-created archives
                File parent = newFile.getParentFile();
                if (!parent.isDirectory() && !parent.mkdirs()) {
                    throw new IOException("Failed to create directory " + parent);
                }

                // write file content
                FileOutputStream fos = new FileOutputStream(newFile);
                int len;
                while ((len = zis.read(buffer)) > 0) {
                    fos.write(buffer, 0, len);
                }
                fos.close();
            }
            zipEntry = zis.getNextEntry();
        }
        zis.closeEntry();
        zis.close();

        System.out.println("Test resources ready!");
    }


    @Test
    public void example() {
        String jobName = "mlp_inference";

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("1");

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        float[] image = readImage("mnist_test/test_set/00000000_7.png");
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("Input_14", imageTensor);
        tensorMap.put("Input_15", tagTensor);

        Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
        for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
            Tensor resTensor = entry.getValue();
            float[] resFloatArray = resTensor.getDataAsFloatArray();
        }
        inferenceSession.close();
    }

    /**
     * run test on cpu
     */
    @Test
    public void cpuTest() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";
        Option option = new Option();
        option.setDeviceTag("cpu")
                .setControlPort(11245)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");

        basicTest(jobName, option);
    }

    /**
     * run test on gpu
     */
    @Test
    public void gpuTest() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");

        basicTest(jobName, option);
    }

    /**
     * open a session and close it, do it again.
     */
    @Test
    public void reopenTest() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");
        basicTest(jobName, option);
        basicTest(jobName, option);

        option.setDeviceTag("cpu");
        basicTest(jobName, option);
        basicTest(jobName, option);
    }

    /**
     * The basic test is tested on models/2, which has signature
     * test on models/1, there is no signature
     */
    @Test
    public void noSignatureTest() {
        String jobName = "mlp_inference";

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("1");

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();

        for (int i = 0; i < IMAGE_FILES.length; i++) {
            String imageFile = IMAGE_FILES[i];
            float[] image = readImage("mnist_test/test_set/" + imageFile);
            Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
            Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
            Map<String, Tensor> tensorMap = new HashMap<>();
            tensorMap.put("Input_14", imageTensor);
            tensorMap.put("Input_15", tagTensor);

            Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
            for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
                Tensor resTensor = entry.getValue();
                float[] resFloatArray = resTensor.getDataAsFloatArray();
                assertArrayEquals(IMAGE_MODEL1_LOGITS[i], resFloatArray, DELTA);
            }
        }
        inferenceSession.close();
    }

    /**
     * test batching, test on model 3
     */
    @Test
    public void batchingTest() {
        String jobName = "mlp_inference";
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("3")
                .setBatchSize(10);

        FloatBuffer floatBuffer = FloatBuffer.allocate(10 * 28 * 28);
        for (String imageFile : IMAGE_FILES) {
            float[] image = readImage("mnist_test/test_set/" + imageFile);
            floatBuffer.put(image);
        }
        Tensor imageTensor = Tensor.fromBlob(floatBuffer.array(), new long[]{ 10, 1, 28, 28 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("image", imageTensor);

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
        for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
            Tensor resTensor = entry.getValue();
            float[] resFloatArray = resTensor.getDataAsFloatArray();
            assertArrayEquals(IMAGE_MODEL3_LOGITS, resFloatArray, DELTA);
        }
        inferenceSession.close();
    }

    /**
     * Some field of Option must be given: deviceTag, savedModelDir, modelVersion, controlPort
     */
    @Test(expected = RuntimeException.class)
    public void optionCompletenessTest0() {
        Option option = new Option();
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        // the session is not opened and the code will not reach here
        inferenceSession.close();
    }

    @Test(expected = RuntimeException.class)
    public void optionCompletenessTest1() {
        Option option = new Option();
        option.setDeviceTag("gpu");
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.close();
    }

    @Test(expected = RuntimeException.class)
    public void optionCompletenessTest2() {
        Option option = new Option();
        option.setDeviceTag("gpu").setControlPort(32321);
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.close();
    }

    @Test(expected = RuntimeException.class)
    public void optionCompletenessTest3() {
        Option option = new Option();
        option.setDeviceTag("gpu").setControlPort(32321).setSavedModelDir("mnist_test/models");
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.close();
    }

    @Test
    public void optionCompletenessTest4() {
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(32321)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("3");
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.close();
    }

    /**
     * If the model proto files not exist, libprotobuf will report error
     * and the user will hard to figure out what the problem is
     */
    @Test(expected = RuntimeException.class)
    public void noModelProtoTest0() {
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(32321)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("5");
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.close();
    }

    @Test(expected = RuntimeException.class)
    public void noModelProtoTest1() {
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(32321)
                .setSavedModelDir("mnist_test/models_not_exist_dir")
                .setModelVersion("1");
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.close();
    }

    /**
     * If there is no checkpoint, the program will also work, but the result is false.
     * So if there is no checkpoint, we need to tell the user by raising a RuntimeException
     */
    @Test
    public void noCheckpointDirTest() {
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(32321)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("4");
        InferenceSession inferenceSession = new InferenceSession(option);
        try {
            inferenceSession.open();
            // shouldn't be here
            fail();
        }
        catch (RuntimeException e) {
            assertTrue(true);
        }
        finally {
            // the session is opened, it should be closed
            inferenceSession.close();
        }
    }

    /**
     * The session has internal status, the user MUST follow the procedure:
     * open -> run -> close
     */
    @Test
    public void sessionStatusCheckTest() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";
        float[] image = readImage("mnist_test/test_set/" + IMAGE_FILES[0]);
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("image", imageTensor);

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");

        InferenceSession inferenceSession = new InferenceSession(option);

        // call close
        try {
            inferenceSession.close();
            fail();
        }
        catch (RuntimeException e) {
            assertTrue(true);
        }

        // call run
        try {
            inferenceSession.run(jobName, tensorMap);
            fail();
        }
        catch (RuntimeException e) {
            assertTrue(true);
        }

        inferenceSession.open();
        inferenceSession.run(jobName, tensorMap);
        inferenceSession.close();

        // call run
        try {
            inferenceSession.run(jobName, tensorMap);
            fail();
        }
        catch (RuntimeException e) {
            assertTrue(true);
        }
    }

    /**
     * reuse one session
     */
    @Test
    public void reuseSessionTest() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";
        float[] image = readImage("mnist_test/test_set/" + IMAGE_FILES[0]);
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("image", imageTensor);

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        inferenceSession.run(jobName, tensorMap);
        inferenceSession.close();

        inferenceSession.open();
        inferenceSession.run(jobName, tensorMap);
        inferenceSession.close();

        inferenceSession.open();
        inferenceSession.close();
    }

    class ServingThread extends Thread {

        private final String jobName;
        private final InferenceSession session;

        public ServingThread(String jobName, InferenceSession session) {
            this.jobName = jobName;
            this.session = session;
        }

        @Override
        public void run() {
            for (int i = 0; i < IMAGE_FILES.length; i++) {
//                long curTime = System.currentTimeMillis();

                String imageFile = IMAGE_FILES[i];
                float[] image = readImage("mnist_test/test_set/" + imageFile);
                Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
                Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
                Map<String, Tensor> tensorMap = new HashMap<>();
                tensorMap.put("Input_14", imageTensor);
                tensorMap.put("Input_15", tagTensor);

                Map<String, Tensor> resultMap = session.run(jobName, tensorMap);
                for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
                    Tensor resTensor = entry.getValue();
                    float[] resFloatArray = resTensor.getDataAsFloatArray();
                    assertArrayEquals(IMAGE_MODEL1_LOGITS[i], resFloatArray, DELTA);
                }

//                System.out.println(getId() + ": " + (System.currentTimeMillis() - curTime) + "ms");
            }
        }
    }

    /**
     * multi-thread test
     */
    @Test
    public void multiThreadTest() throws InterruptedException {
        String jobName = "mlp_inference";

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir("mnist_test/models")
                .setModelVersion("1");

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();

        Thread th0 = new ServingThread(jobName, inferenceSession);
        Thread th1 = new ServingThread(jobName, inferenceSession);
        Thread th2 = new ServingThread(jobName, inferenceSession);
        Thread th3 = new ServingThread(jobName, inferenceSession);
        Thread th4 = new ServingThread(jobName, inferenceSession);
        Thread th5 = new ServingThread(jobName, inferenceSession);

        th0.start();
        th1.start();
        th2.start();
        th3.start();
        th4.start();
        th5.start();

        th0.join();
        th1.join();
        th2.join();
        th3.join();
        th4.join();
        th5.join();

        inferenceSession.close();
    }

    @Test
    public void predictImages() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");
        makePredict(jobName, option);
    }

    public void makePredict(String jobName, Option option) {
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();

        for (String imageFile : IMAGE_FILES) {
            float[] image = readImage("mnist_test/test_set/" + imageFile);
            Tensor imageTensor = Tensor.fromBlob(image, new long[]{1, 1, 28, 28});
            Map<String, Tensor> tensorMap = new HashMap<>();
            tensorMap.put("image", imageTensor);

            Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
            for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
                Tensor resTensor = entry.getValue();
                float[] resFloatArray = resTensor.getDataAsFloatArray();
                int digit = 0;
                float maxLogit = resFloatArray[0];
                for (int candidate = 1; candidate < resFloatArray.length; candidate++) {
                    if (resFloatArray[candidate] > maxLogit) {
                        digit = candidate;
                        maxLogit = resFloatArray[candidate];
                    }
                }
                System.out.println("image file: " + imageFile + " predict: " + digit);
            }
        }
        inferenceSession.close();
    }

    private void basicTest(String jobName, Option option) {
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();

        for (int i = 0; i < IMAGE_FILES.length; i++) {
//            long curTime = System.currentTimeMillis();
            String imageFile = IMAGE_FILES[i];
            float[] image = readImage("mnist_test/test_set/" + imageFile);
            Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
            Map<String, Tensor> tensorMap = new HashMap<>();
            tensorMap.put("image", imageTensor);

            Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
            for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
                Tensor resTensor = entry.getValue();
                float[] resFloatArray = resTensor.getDataAsFloatArray();
                assertArrayEquals(IMAGE_MODEL2_LOGITS[i], resFloatArray, DELTA);
            }
//            System.out.println(System.currentTimeMillis() - curTime);
        }
        inferenceSession.close();
    }

    private float[] readImage(String filePath) {
        File file = new File(filePath);
        BufferedImage image = null;
        try {
            image = ImageIO.read(file);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        assert image != null;

        int width = image.getWidth();
        int height = image.getHeight();
        Raster raster = image.getRaster();
        float[] pixels = new float[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                // Caution: transform the image
                pixels[i * width + j] = (raster.getSample(j, i, 0) - 128.0f) / 255.0f;
            }
        }
        return pixels;
    }

}
