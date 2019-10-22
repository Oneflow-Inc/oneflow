import cv2
import copy
import numpy as np
import pycocotools.mask as mask_utils
#from maskrcnn_benchmark.utils import cv2_util

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


""" ABSTRACT
Segmentations come in either:
1) Binary masks
2) Polygons

Binary masks can be represented in a contiguous array
and operations can be carried out more efficiently,
therefore BinaryMaskList handles them together.

Polygons are handled separately for each instance,
by PolygonInstance and instances are handled by
PolygonList.

SegmentationList is supposed to represent both,
therefore it wraps the functions of BinaryMaskList
and PolygonList to make it transparent.
"""


class BinaryMaskList(object):
    """
    This class handles binary masks for all objects in the image
    """

    def __init__(self, masks, size):
        """
            Arguments:
                masks: Either numpy.ndarray of [num_instances, H, W]
                    or list of numpy.ndarray of [H, W] with num_instances elems,
                    or RLE (Run Length Encoding) - interpreted as list of dicts,
                    or BinaryMaskList.
                size: absolute image size, width first

            After initialization, a hard copy will be made, to leave the
            initializing source data intact.
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2

        if isinstance(masks, np.ndarray):
            # The raw data representation is passed as argument
            masks = masks.clone()
        elif isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                masks = np.empty([0, size[1], size[0]])  # num_instances = 0!
            elif isinstance(masks[0], np.ndarray):
                masks = np.stack(masks, dim=0).clone()
            elif isinstance(masks[0], dict) and "counts" in masks[0]:
                if(isinstance(masks[0]["counts"], (list, tuple))):
                    masks = mask_utils.frPyObjects(masks, size[1], size[0])
                # RLE interpretation
                rle_sizes = [tuple(inst["size"]) for inst in masks]

                masks = mask_utils.decode(masks)  # [h, w, n]
                masks = np.transpose(np.array(masks), (2, 0, 1))  # [n, h, w]

                assert rle_sizes.count(rle_sizes[0]) == len(rle_sizes), (
                    "All the sizes must be the same size: %s" % rle_sizes
                )

                # in RLE, height come first in "size"
                rle_height, rle_width = rle_sizes[0]
                assert masks.shape[1] == rle_height
                assert masks.shape[2] == rle_width

                width, height = size
                if width != rle_width or height != rle_height:
                    masks = cv2.resize(
                        masks[None].float(),
                        dsize=(height, width),
                        interpolation=cv2.INTER_LINEAR
                    )[0]
            else:
                RuntimeError(
                    "Type of `masks[0]` could not be interpreted: %s"
                    % type(masks)
                )
        elif isinstance(masks, BinaryMaskList):
            # just hard copy the BinaryMaskList instance's underlying data
            masks = masks.masks.clone()
        else:
            RuntimeError(
                "Type of `masks` argument could not be interpreted:%s"
                % type(masks)
            )

        if len(masks.shape) == 2:
            # if only a single instance mask is passed
            masks = masks[None]

        assert len(masks.shape) == 3
        assert masks.shape[1] == size[1], "%s != %s" % (masks.shape[1], size[1])
        assert masks.shape[2] == size[0], "%s != %s" % (masks.shape[2], size[0])

        self.masks = masks
        self.size = tuple(size)

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = self.masks.flip(dim)
        return BinaryMaskList(flipped_masks, self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, np.ndarray)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_masks = self.masks[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return BinaryMaskList(cropped_masks, cropped_size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_masks = cv2.resize(
            self.masks[None].float(),
            (height, width),
            interpolation=cv2.INTER_LINEAR,
        )[0]
        resized_size = width, height
        return BinaryMaskList(resized_masks, resized_size)

    def convert_to_polygon(self):
        if self.masks.numel() == 0:
            return PolygonList([], self.size)

        contours = self._findContours()
        return PolygonList(contours, self.size)

    def to(self, *args, **kwargs):
        return self

    def _findContours(self):
        raise "to be confirm"
        contours = []
        masks = self.masks.detach().numpy()
        for mask in masks:
            mask = cv2.UMat(mask)
            contour, hierarchy = cv2_util.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )

            reshaped_contour = []
            for entity in contour:
                assert len(entity.shape) == 3
                assert (
                    entity.shape[1] == 1
                ), "Hierarchical contours are not allowed"
                reshaped_contour.append(entity.reshape(-1).tolist())
            contours.append(reshaped_contour)
        return contours

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        if self.masks.numel() == 0:
            raise RuntimeError("Indexing empty BinaryMaskList")
        return BinaryMaskList(self.masks[index], self.size)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class PolygonInstance(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size):
        """
            Arguments:
                a list of lists of numbers.
                The first level refers to all the polygons that compose the
                object, and the second level to the polygon coordinates.
        """
        if isinstance(polygons, (list, tuple)):
            valid_polygons = []
            for p in polygons:
                p = np.array(p, dtype=np.float32)
                if len(p) >= 6:  # 3 * 2 coordinates
                    valid_polygons.append(p)
            polygons = valid_polygons

        elif isinstance(polygons, PolygonInstance):
            polygons = copy.copy(polygons.polygons)

        else:
            RuntimeError(
                "Type of argument `polygons` is not allowed:%s"
                % (type(polygons))
            )

        """ This crashes the training way too many times...
        for p in polygons:
            assert p[::2].min() >= 0
            assert p[::2].max() < size[0]
            assert p[1::2].min() >= 0
            assert p[1::2].max() , size[1]
        """

        self.polygons = polygons
        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return PolygonInstance(flipped_polygons, size=self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, np.ndarray)), str(type(box))

        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = map(float, box)

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        w, h = xmax - xmin, ymax - ymin

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - xmin  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - ymin  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return PolygonInstance(cropped_polygons, size=(w, h))

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size

        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(size, self.size)
        )

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return PolygonInstance(scaled_polys, size)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return PolygonInstance(scaled_polygons, size=size)

    def convert_to_binarymask(self):
        width, height = self.size
        # formatting for COCO PythonAPI
        polygons = [p.numpy() for p in self.polygons]
        rles = mask_utils.frPyObjects(polygons, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
        return mask

    def __len__(self):
        return len(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_groups={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class PolygonList(object):
    """
    This class handles PolygonInstances for all objects in the image
    """

    def __init__(self, polygons, size):
        """
        Arguments:
            polygons:
                a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.

                OR

                a list of PolygonInstances.

                OR

                a PolygonList

            size: absolute image size

        """
        if isinstance(polygons, (list, tuple)):
            if len(polygons) == 0:
                polygons = [[[]]]
            if isinstance(polygons[0], (list, tuple)):
                assert isinstance(polygons[0][0], (list, tuple)), str(
                    type(polygons[0][0])
                )
            else:
                assert isinstance(polygons[0], PolygonInstance), str(
                    type(polygons[0])
                )

        elif isinstance(polygons, PolygonList):
            size = polygons.size
            polygons = polygons.polygons

        else:
            RuntimeError(
                "Type of argument `polygons` is not allowed:%s"
                % (type(polygons))
            )

        assert isinstance(size, (list, tuple)), str(type(size))

        self.polygons = []
        for p in polygons:
            p = PolygonInstance(p, size)
            if len(p) > 0:
                self.polygons.append(p)

        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        for polygon in self.polygons:
            flipped_polygons.append(polygon.transpose(method))

        return PolygonList(flipped_polygons, size=self.size)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_polygons = []
        for polygon in self.polygons:
            cropped_polygons.append(polygon.crop(box))

        cropped_size = w, h
        return PolygonList(cropped_polygons, cropped_size)

    def resize(self, size):
        resized_polygons = []
        for polygon in self.polygons:
            resized_polygons.append(polygon.resize(size))

        resized_size = size
        return PolygonList(resized_polygons, resized_size)

    def to(self, *args, **kwargs):
        return self

    def convert_to_binarymask(self):
        if len(self) > 0:
            masks = np.stack([p.convert_to_binarymask() for p in self.polygons])
        else:
            size = self.size
            masks = np.empty([0, size[1], size[0]], dtype=np.uint8)

        return BinaryMaskList(masks, size=self.size)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, item):
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, np.ndarray) and item.dtype == np.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return PolygonList(selected_polygons, size=self.size)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class SegmentationMask(object):

    """
    This class stores the segmentations for all objects in the image.
    It wraps BinaryMaskList and PolygonList conveniently.
    """

    def __init__(self, instances, size, mode="poly"):
        """
        Arguments:
            instances: two types
                (1) polygon
                (2) binary mask
            size: (width, height)
            mode: 'poly', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        if isinstance(size[0], np.ndarray):
            assert isinstance(size[1], np.ndarray)
            size = size[0].item(), size[1].item()

        assert isinstance(size[0], (int, float))
        assert isinstance(size[1], (int, float))

        if mode == "poly":
            self.instances = PolygonList(instances, size)
        elif mode == "mask":
            self.instances = BinaryMaskList(instances, size)
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        self.mode = mode
        self.size = tuple(size)

    def transpose(self, method):
        flipped_instances = self.instances.transpose(method)
        return SegmentationMask(flipped_instances, self.size, self.mode)

    def crop(self, box):
        cropped_instances = self.instances.crop(box)
        cropped_size = cropped_instances.size
        return SegmentationMask(cropped_instances, cropped_size, self.mode)

    def resize(self, size, *args, **kwargs):
        resized_instances = self.instances.resize(size)
        resized_size = size
        return SegmentationMask(resized_instances, resized_size, self.mode)

    def to(self, *args, **kwargs):
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self

        if mode == "poly":
            converted_instances = self.instances.convert_to_polygon()
        elif mode == "mask":
            converted_instances = self.instances.convert_to_binarymask()
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        return SegmentationMask(converted_instances, self.size, mode)

    def get_mask_tensor(self):
        instances = self.instances
        if self.mode == "poly":
            instances = instances.convert_to_binarymask()
        # If there is only 1 instance
        return instances.masks.squeeze(0)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        selected_instances = self.instances.__getitem__(item)
        return SegmentationMask(selected_instances, self.size, self.mode)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_segmentation = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_segmentation
        raise StopIteration()

    next = __next__  # Python 2 compatibility

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.instances))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
