package org.oneflow;

import org.junit.Test;

import static org.junit.Assert.*;

public class OptionTest {

    @Test
    public void nullTest() {
        Option option = new Option();
        assertNull(option.getBatchSize());
        assertNull(option.getControlPort());
        assertNull(option.getDeviceTag());
        assertNull(option.getGraphName());
        assertNull(option.getMetaFileBaseName());
        assertNull(option.getMirroredView());
        assertNull(option.getModelVersion());
        assertNull(option.getSavedModelDir());
        assertNull(option.getSignatureName());
    }

    @Test
    public void buildTest() {
        Option option = new Option();
        option.setDeviceTag("gpu")
                .setControlPort(12345)
                .setMirroredView(false)
                .setModelVersion("1")
                .setSavedModelDir("./models")
                .setSignatureName("mlp");

        assertEquals(option.getDeviceTag(), "gpu");
        assertNotNull(option.getControlPort());
        assertEquals((int) option.getControlPort(), 12345);
        assertFalse(option.getMirroredView());
        assertNotNull(option.getModelVersion());
        assertEquals(option.getModelVersion(), "1");
        assertEquals(option.getSavedModelDir(), "./models");
        assertEquals(option.getSignatureName(), "mlp");

        assertNull(option.getBatchSize());
        assertNull(option.getGraphName());
        assertNull(option.getMetaFileBaseName());
    }

}
