class Matcher(object):
    def __init__(self, fg_iou_threshold, bg_iou_threshold):
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold

    # anchors: [M, 4]
    # gt_boxes: [G, 4]
    def build(self, anchors, gt_boxes, allow_low_quality_matches):
        # iou_matrix: [M, G]
        iou_matrix = flow.detection.calc_iou_matrix(anchors, gt_boxes)
        anchor_indices = flow.math.top_k(iou_matrix, k=1)
        # anchor_matched_iou: [M]
        anchor_matched_iou = flow.squeeze(
            flow.gather(params=iou_matrix, indices=anchor_indices, batch_dims=1),
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
        mask = flow.logical_and(
            anchor_matched_iou
            < flow.constant(value=self.fg_iou_threshold, dtype=flow.float, shape=[1]),
            anchor_matched_iou
            >= flow.constant(value=self.bg_iou_threshold, dtype=flow.float, shape=[1]),
        )
        matched_indices = flow.where(
            mask, flow.constant_like(matched_indices, float(-2)), matched_indices
        )

        if allow_low_quality_matches:
            # iou_matrix_trans: [G, M]
            iou_matrix_trans = flow.detection.calc_iou_matrix(gt_boxes, anchors)
            # gt_matched_iou: [G, 1]
            gt_matched_iou = flow.batch_gather(
                params=iou_matrix_trans,
                indices=flow.math.top_k(iou_matrix_trans, k=1),
                batch_dims=1,
            )
            update_indices = flow.slice(
                flow.local_nonzero(iou_matrix_trans == gt_matched_iou),
                dim_slice_conf=[{"start": 1, "end": 2, "stride": 1}],
            )
            matched_indices = flow.local_scatter_nd_update(
                matched_indices,
                update_indices,
                flow.local_gather(squeezed_anchor_indices, update_indices),
            )

        return matched_indices
