package org.oneflow;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class InferenceSessionTest {

    @BeforeClass
    public static void downloadModel() {
        System.out.println("download resources");
    }

    @Test
    public void main() {
        String jobName = "mlp_inference";
        String savedModelDir = "./models";
        float[] image = readImage("./7.png");
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("image", imageTensor);

        // Option
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(11245)
                .setMirroredView(false)
                .setSavedModelDir(savedModelDir);

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
