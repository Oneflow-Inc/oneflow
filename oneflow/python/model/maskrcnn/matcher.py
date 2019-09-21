class Matcher(object):
    def __init__(self, dl_net, cfg, fg_iou_threshold, bg_iou_threshold, img_idx):
        self.cfg = cfg
        self.fg_iou_threshold = fg_iou_threshold
        self.bg_iou_threshold = bg_iou_threshold
        self.img_idx = img_idx

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
        mask = self.less_than(
            anchor_matched_iou,
            self.constant_scalar(iou_matrix, float(self.bg_iou_threshold)),
        )
        squeezed_anchor_indices = self.squeeze(anchor_indices, axis=[1])
        matched_indices = self.where(
            mask,
            self.constant_like(squeezed_anchor_indices, float(-1)),
            squeezed_anchor_indices,
        )
        mask = self.logical_and(
            self.less_than(
                anchor_matched_iou,
                self.constant_scalar(iou_matrix, float(self.fg_iou_threshold)),
            ),
            self.greater_equal(
                anchor_matched_iou,
                self.constant_scalar(iou_matrix, float(self.bg_iou_threshold)),
            ),
        )
        matched_indices = self.where(
            mask, self.constant_like(matched_indices, float(-2)), matched_indices
        )

        if allow_low_quality_matches:
            # iou_matrix_trans: [G, M]
            iou_matrix_trans = self.calc_iou_matrix(gt_boxes, anchors)
            # gt_matched_iou: [G, 1]
            gt_matched_iou = self.batch_gather(
                iou_matrix_trans, self.top_k(iou_matrix_trans, k=1)
            )
            update_indices = self.dl_net.Slice(
                self.nonzero(self.equal(iou_matrix_trans, gt_matched_iou)),
                dim_slice_conf=[{"start": 1, "end": 2, "stride": 1}],
                name="slice_img_{}".format(self.img_idx),
            )
            matched_indices = self.scatter_nd_update(
                matched_indices,
                update_indices,
                self.gather(squeezed_anchor_indices, update_indices),
            )

        return matched_indices
