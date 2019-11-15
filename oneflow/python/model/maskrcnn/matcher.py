import oneflow as flow
import operator
from functools import reduce


class Matcher(object):
    def __init__(self, fg_iou_threshold, bg_iou_threshold):
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold

    # anchors: [M, 4]
    # gt_boxes: [G, 4]
    def build(self, anchors, gt_boxes, allow_low_quality_matches):
        # CHECK_POINT: iou_matrix: [M, G]
        iou_matrix = flow.detection.calc_iou_matrix(anchors, gt_boxes)
        # TODO: do not need expand_dims here
        anchor_indices = flow.expand_dims(flow.math.argmax(iou_matrix), axis=1)
        # anchor_matched_iou: [M]
        anchor_matched_iou = flow.squeeze(
            flow.gather(
                params=iou_matrix, indices=anchor_indices, batch_dims=1
            ),
            axis=[1],
        )
        mask = anchor_matched_iou < flow.constant(
            float(self.bg_iou_threshold), dtype=flow.float, shape=[1]
        )
        squeezed_anchor_indices = flow.squeeze(anchor_indices, axis=[1])
        matched_indices = flow.where(
            mask,
            flow.constant_like(squeezed_anchor_indices, float(-1)),
            squeezed_anchor_indices,
        )
        mask = flow.math.logical_and(
            anchor_matched_iou
            < flow.constant(
                value=self.fg_iou_threshold, dtype=flow.float, shape=[1]
            ),
            anchor_matched_iou
            >= flow.constant(
                value=self.bg_iou_threshold, dtype=flow.float, shape=[1]
            ),
        )
        matched_indices = flow.where(
            mask,
            flow.constant_like(matched_indices, float(-2)),
            matched_indices,
        )

        if allow_low_quality_matches:
            # iou_matrix_trans: [G, M]
            iou_matrix_trans = flow.detection.calc_iou_matrix(gt_boxes, anchors)
            # gt_matched_iou: [G, 1]
            gt_matched_iou = flow.gather(
                params=iou_matrix_trans,
                # TODO: do not need expand_dims here
                indices=flow.expand_dims(
                    flow.math.argmax(iou_matrix_trans), axis=1
                ),
                batch_dims=1,
            )
            box_max_gt = flow.local_nonzero(iou_matrix_trans == gt_matched_iou)
            update_indices = flow.cast(
                flow.matmul(
                    flow.cast(box_max_gt, dtype=flow.float32),
                    flow.concat(
                        [
                            flow.constant(0, shape=(1, 1), dtype=flow.float32),
                            flow.constant(1, shape=(1, 1), dtype=flow.float32),
                        ],
                        axis=0,
                    ),
                ),
                dtype=box_max_gt.dtype,
            )
            matched_indices = flow.local_scatter_nd_update(
                matched_indices,
                update_indices,
                flow.local_gather(squeezed_anchor_indices, update_indices),
            )

        return matched_indices
