package org.oneflow;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class InferenceSessionTest {

    @Test
    public void testInference() {
        String jobName = "mlp_inference";
        String savedModelDir = "/home/percent1/tmp/target/models";
        float[] image = readImage("/home/percent1/tmp/target/7.png");
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("Input_14", imageTensor);
        tensorMap.put("Input_15", tagTensor);

        InferenceSession inferenceSession = new InferenceSession(8888);
        inferenceSession.open();
        inferenceSession.loadModel(savedModelDir);
        inferenceSession.launch();

        Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
        inferenceSession.close();

        // assert
        float[] vector = resultMap.get("Return_17").getDataAsFloatArray();
        assertEquals(10, vector.length);
        float[] expectedVector = { -129.57167f, -89.084816f, -139.21355f , -103.455025f, -9.179366f,
                -69.568474f, -133.39594f,  -16.204329f, -114.90876f,  -47.933548f };
        float delta = 0.0001f;
        for (int i = 0; i < 10; i++) {
            assertEquals(expectedVector[i], vector[i], delta);
        }
    }

    public static float[] readImage(String filePath) {
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
