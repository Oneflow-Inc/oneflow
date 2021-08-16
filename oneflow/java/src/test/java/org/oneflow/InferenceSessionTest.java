package org.oneflow;

import org.junit.BeforeClass;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.math.BigInteger;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import static org.junit.Assert.*;

public class InferenceSessionTest {

    private final static String ZIP_FILE = "mnist_test.zip";
    private final static String ZIP_MD5 = "67a061f87d034d4d53ec572f512449ab";
    private final static String ZIP_URL =
            "https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/java-serving/mnist_test.zip";
    private final static String UNZIP_DIR = "mnist_test";
    private final static int BUFFER_SIZE = 1024;

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
                int read = 0;
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
            File newFile = new File(unzipDir, String.valueOf(zipEntry));
            if (zipEntry.isDirectory()) {
                if (!newFile.isDirectory() && !newFile.mkdirs()) {
                    throw new IOException("Failed to create directory " + newFile);
                }
            } else {
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
    public void main() {
        String jobName = "mlp_inference";
        String savedModelDir = "mnist_test/models";
        float[] image = readImage("mnist_test/test_set/00000000_7.png");
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("image", imageTensor);

        // Option
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setMirroredView(false)
                .setSavedModelDir(savedModelDir)
                .setModelVersion("2");

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();

        Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
        for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
            Tensor resTensor = entry.getValue();
            float[] resFloatArray = resTensor.getDataAsFloatArray();
            System.out.println(Arrays.toString(resFloatArray));
        }

        int forwardTimes = 10;
        long curTime = System.currentTimeMillis();
        for (int i = 0; i < forwardTimes; i++) {
            resultMap = inferenceSession.run(jobName, tensorMap);
            for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
                Tensor resTensor = entry.getValue();
                resTensor.getDataAsFloatArray();
            }
        }
        System.out.printf("It takes %fs to forward %d times\n",
                (System.currentTimeMillis() - curTime) / 1000.0f,
                forwardTimes);

        inferenceSession.close();

        // assert
        float[] vector = resultMap.get("output").getDataAsFloatArray();
        assertEquals(vector.length, 10);
        float[] expectedVector = { -130.93361f, -72.18875f, -165.4578f, -114.832054f, -21.068695f,
                -88.18618f, -151.45547f, 19.964626f, -142.91219f, -43.097794f};
        float delta = 0.0001f;
        for (int i = 0; i < 10; i++) {
            assertEquals(expectedVector[i], vector[i], delta);
        }
    }

    @Test
    public void reopenTest() {

    }

    public float[] readImage(String filePath) {
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
