# This file takes partial of the implementation from NVIDIA's webdataset at here:
# https://github.com/tmbdev/webdataset/blob/master/webdataset/autodecode.py

import io
import os
import pickle
import re
import tempfile

import json
import oneflow as flow


################################################################
# handle basic datatypes
################################################################


def basichandlers(key, data):

    extension = re.sub(r".*[.]", "", key)

    if extension in "txt text transcript":
        return data.decode("utf-8")

    if extension in "cls cls2 class count index inx id".split():
        try:
            return int(data)
        except ValueError:
            return None

    if extension in "json jsn":
        return json.loads(data)

    if extension in "pyd pickle".split():
        return pickle.loads(data)

    if extension in "pt".split():
        stream = io.BytesIO(data)
        return flow.load(stream)

    # if extension in "ten tb".split():
    #     from . import tenbin
    #     return tenbin.decode_buffer(data)

    # if extension in "mp msgpack msg".split():
    #     import msgpack
    #     return msgpack.unpackb(data)

    return None


################################################################
# handle images
################################################################

imagespecs = {
    "l8": ("numpy", "uint8", "l"),
    "rgb8": ("numpy", "uint8", "rgb"),
    "rgba8": ("numpy", "uint8", "rgba"),
    "l": ("numpy", "float", "l"),
    "rgb": ("numpy", "float", "rgb"),
    "rgba": ("numpy", "float", "rgba"),
    "flowl8": ("flow", "uint8", "l"),
    "flowrgb8": ("flow", "uint8", "rgb"),
    "flowrgba8": ("flow", "uint8", "rgba"),
    "flowl": ("flow", "float", "l"),
    "flowrgb": ("flow", "float", "rgb"),
    "flow": ("flow", "float", "rgb"),
    "flowrgba": ("flow", "float", "rgba"),
    "pill": ("pil", None, "l"),
    "pil": ("pil", None, "rgb"),
    "pilrgb": ("pil", None, "rgb"),
    "pilrgba": ("pil", None, "rgba"),
}

def handle_extension(extensions, f):
    """
    Returns a decoder handler function for the list of extensions.
    Extensions can be a space separated list of extensions.
    Extensions can contain dots, in which case the corresponding number
    of extension components must be present in the key given to f.
    Comparisons are case insensitive.
    Examples:
    handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
    handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    """

    extensions = extensions.lower().split()

    def g(key, data):
        extension = key.lower().split(".")

        for target in extensions:
            target = target.split(".")
            if len(target) > len(extension):
                continue

            if extension[-len(target):] == target:
                return f(data)
            return None
    return g


class ImageHandler:
    """
    Decode image data using the given `imagespec`.
    The `imagespec` specifies whether the image is decoded
    to numpy/flow/pi, decoded to uint8/float, and decoded
    to l/rgb/rgba:

    - l8: numpy uint8 l
    - rgb8: numpy uint8 rgb
    - rgba8: numpy uint8 rgba
    - l: numpy float l
    - rgb: numpy float rgb
    - rgba: numpy float rgba
    - flowl8: flow uint8 l
    - flowrgb8: flow uint8 rgb
    - flowrgba8: flow uint8 rgba
    - flowl: flow float l
    - flowrgb: flow float rgb
    - flow: flow float rgb
    - flowrgba: flow float rgba
    - pill: pil None l
    - pil: pil None rgb
    - pilrgb: pil None rgb
    - pilrgba: pil None rgba
    """
    def __init__(self, imagespec):
        assert imagespec in list(imagespecs.keys()), "unknown image specification: {}".format(imagespec)
        self.imagespec = imagespec.lower()

    def __call__(self, key, data):
        extension = re.sub(r".*[.]", "", key)
        if extension.lower() not in "jpg jpeg png ppm pgm pbm pnm".split():
            return None

        try:
            import numpy as np
        except ImportError as e:
            raise ModuleNotFoundError("Package `numpy` is required to be installed for default image decoder."
                                      "Please use `pip install numpy` to install the package")

        try:
            import PIL.Image
        except ImportError as e:
            raise ModuleNotFoundError("Package `PIL` is required to be installed for default image decoder."
                                      "Please use `pip install Pillow` to install the package")

        imagespec = self.imagespec
        atype, etype, mode = imagespecs[imagespec]

        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
            if atype == "pil":
                return img
            elif atype == "numpy":
                result = np.asarray(img)
                assert result.dtype == np.uint8, "numpy image array should be type uint8, but got {}".format(result.dtype)
                if etype == "uint8":
                    return result
                else:
                    return result.astype("f") / 255.0
            elif atype == "flow":
                result = np.asarray(img)
                assert result.dtype == np.uint8, "numpy image array should be type uint8, but got {}".format(result.dtype)

                if etype == "uint8":
                    result = np.array(result.transpose(2, 0, 1))
                    return flow.tensor(result)
                else:
                    result = np.array(result.transpose(2, 0, 1))
                    return flow.tensor(result) / 255.0
            return None

def imagehandler(imagespec):
    return ImageHandler(imagespec)


################################################################
# flow video
################################################################


def flow_video(key, data):
    extension = re.sub(r".*[.]", "", key)
    if extension not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
        return None

    try:
        import flowvision.io
    except ImportError as e:
        raise ModuleNotFoundError("Package `flowvision` is required to be installed for default video file loader."
                                  "Please use `pip install flowvision` or `conda install flowvision -c oneflow`"
                                  "to install the package")

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return flowvision.io.read_video(fname)


################################################################
# flowaudio
################################################################


def flow_audio(key, data):
    extension = re.sub(r".*[.]", "", key)
    if extension not in ["flac", "mp3", "sox", "wav", "m4a", "ogg", "wma"]:
        return None

    try:
        import flowaudio  # type: ignore
    except ImportError as e:
        raise ModuleNotFoundError("Package `flowaudio` is required to be installed for default audio file loader."
                                  "Please use `pip install flowaudio` or `conda install flowaudio -c oneflow`"
                                  "to install the package")

    with tempfile.TemporaryDirectory() as dirname:
        fname = os.path.join(dirname, f"file.{extension}")
        with open(fname, "wb") as stream:
            stream.write(data)
            return flowaudio.load(fname)



################################################################
# a sample decoder
################################################################


class Decoder:
    """
    Decode key/data sets using a list of handlers.
    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, handlers):
        self.handlers = handlers

    def add_handler(self, handler):
        if not handler:
            return
        if not self.handlers:
            self.handlers = [handler]
        else:
            self.handlers.append(handler)

    def decode1(self, key, data):
        if not data:
            return data

        # if data is a stream handle, we need to read all the content before decoding
        if isinstance(data, io.BufferedIOBase) or isinstance(data, io.RawIOBase):
            data = data.read()

        for f in self.handlers:
            result = f(key, data)
            if result is not None:
                return result
        return data

    def decode(self, data):
        result = {}
        # single data tuple(pathname, data stream)
        if isinstance(data, tuple):
            data = [data]

        if data is not None:
            for k, v in data:
                # TODO: xinyu, figure out why Nvidia do this?
                if k[0] == "_":
                    if isinstance(v, bytes):
                        v = v.decode("utf-8")
                        result[k] = v
                        continue
                result[k] = self.decode1(k, v)
        return result

    def __call__(self, data):
        return self.decode(data)
