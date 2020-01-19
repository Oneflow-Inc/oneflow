from functools import reduce
import operator

from matcher import Matcher
import oneflow as flow


class BoxHead(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # proposals: list of [R, 4] wrt. images
    # gt_boxes_list: list of [G, 4] wrt. images
    # gt_labels_list: list of [G] wrt. images
    # features: list of [N, C_i, H_i, W_i] wrt. fpn layers
    def build_train(
        self,
        proposals,
        gt_boxes_list,
        gt_labels_list,
        features,
        return_total_pos_inds_elem_cnt=False,
    ):
        with flow.deprecated.variable_scope("roi"):
            # used in box_head
            label_list = []
            proposal_list = []
            bbox_target_list = []
            # used to generate positive proposals for mask_head
            pos_proposal_list = []
            pos_gt_indices_list = []

            for img_idx in range(len(proposals)):
                with flow.deprecated.variable_scope("matcher"):
                    box_head_matcher = Matcher(
                        self.cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
                        self.cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
                    )
                    matched_indices = box_head_matcher.build(
                        proposals[img_idx],
                        gt_boxes_list[img_idx],
                        allow_low_quality_matches=False,
                    )
                pos_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices >= flow.constant_scalar(0, flow.int32)
                    ),
                    axis=[1],
                )
                neg_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices == flow.constant_scalar(-1, flow.int32)
                    ),
                    axis=[1],
                )

                if self.cfg.MODEL.ROI_HEADS.RANDOM_SAMPLE:
                    rand_pos_inds = flow.detection.random_perm_like(pos_inds)
                    rand_neg_inds = flow.detection.random_perm_like(neg_inds)
                    pos_inds = flow.local_gather(pos_inds, rand_pos_inds)
                    neg_inds = flow.local_gather(neg_inds, rand_neg_inds)

                sampled_pos_inds, sampled_neg_inds = flow.detection.pos_neg_sampler(
                    pos_inds,
                    neg_inds,
                    total_subsample_num=self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
                    pos_fraction=self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
                    name="img{}_pos_neg_sampler".format(img_idx),
                )
                sampled_pos_neg_inds = flow.concat(
                    [sampled_pos_inds, sampled_neg_inds], axis=0
                )

                clamped_matched_indices = flow.clip_by_value(
                    t=matched_indices, clip_value_min=0
                )
                proposal_gt_labels = flow.local_gather(
                    gt_labels_list[img_idx], clamped_matched_indices
                )
                proposal_gt_labels = flow.local_scatter_nd_update(
                    proposal_gt_labels,
                    flow.expand_dims(sampled_neg_inds, axis=1),
                    flow.constant_like(sampled_neg_inds, int(0)),
                )
                matched_gt_label = flow.local_gather(
                    proposal_gt_labels, sampled_pos_neg_inds
                )
                label_list.append(matched_gt_label)

                gt_indices = flow.local_gather(
                    clamped_matched_indices, sampled_pos_neg_inds
                )
                pos_gt_indices = flow.local_gather(
                    clamped_matched_indices,
                    sampled_pos_inds,
                    name="img{}_gt_inds".format(img_idx),
                )
                proposal_per_img = flow.local_gather(
                    proposals[img_idx], sampled_pos_neg_inds
                )
                pos_proposal_per_img = flow.local_gather(
                    proposals[img_idx], sampled_pos_inds
                )
                gt_boxes_per_img = flow.local_gather(gt_boxes_list[img_idx], gt_indices)
                (
                    weight_x,
                    weight_y,
                    weight_h,
                    weight_w,
                ) = self.cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
                bbox_target_list.append(
                    flow.detection.box_encode(
                        gt_boxes_per_img,
                        proposal_per_img,
                        regression_weights={
                            "weight_x": weight_x,
                            "weight_y": weight_y,
                            "weight_h": weight_h,
                            "weight_w": weight_w,
                        },
                    )
                )
                proposal_list.append(proposal_per_img)
                pos_proposal_list.append(pos_proposal_per_img)
                pos_gt_indices_list.append(pos_gt_indices)

            proposals = flow.concat(proposal_list, axis=0)
            img_ids = flow.concat(
                flow.detection.extract_piece_slice_id(proposal_list), axis=0
            )
            labels = flow.concat(label_list, axis=0)
            bbox_targets = flow.concat(bbox_target_list, axis=0)

            # box feature extractor
            x = self.box_feature_extractor(proposals, img_ids, features)

            # box predictor
            bbox_regression, cls_logits = self.box_predictor(x)

            # construct cls loss
            box_head_cls_loss = flow.math.reduce_sum(
                flow.nn.sparse_softmax_cross_entropy_with_logits(
                    labels, cls_logits, name="sparse_cross_entropy"
                )
            )
            total_elem_cnt = flow.elem_cnt(labels, dtype=box_head_cls_loss.dtype)
            box_head_cls_loss = flow.math.divide(
                box_head_cls_loss, total_elem_cnt, name="box_head_cls_loss_div"
            )
            # construct bbox loss
            total_pos_inds = flow.squeeze(
                flow.local_nonzero(labels != flow.constant_scalar(int(0), flow.int32)),
                axis=[1],
            )
            # [R, 81, 4]
            pos_bbox_reg = flow.local_gather(bbox_regression, total_pos_inds)
            pos_bbox_reg = flow.dynamic_reshape(pos_bbox_reg, shape=[-1, 81, 4])
            # [R, 1]
            indices = flow.expand_dims(
                flow.local_gather(labels, total_pos_inds), axis=1
            )
            bbox_pred = flow.squeeze(
                flow.gather(params=pos_bbox_reg, indices=indices, batch_dims=1),
                axis=[1],
            )
            bbox_target = flow.local_gather(bbox_targets, total_pos_inds)
            box_head_box_loss = flow.math.divide(
                flow.math.reduce_sum(flow.detection.smooth_l1(bbox_pred, bbox_target)),
                total_elem_cnt,
                name="box_head_box_loss_div",
            )
            ret = (
                box_head_box_loss,
                box_head_cls_loss,
                pos_proposal_list,
                pos_gt_indices_list,
            )
            if return_total_pos_inds_elem_cnt:
                total_pos_inds_elem_cnt = flow.elem_cnt(
                    total_pos_inds, dtype=flow.float
                )
                ret += (total_pos_inds_elem_cnt,)
            else:
                ret += (None,)
            return ret

    # Input:
    # rpn_proposals: list of [R, 4] wrt. images
    # features: list of [N, C_i, H_i, W_i] wrt. fpn layers
    # image_size_list: list of [2,] wrt. images
    # Optput:
    # results: list of (boxes, scores, labels, image_size) wrt. images
    def build_eval(self, rpn_proposals, features, image_size_list):
        with flow.deprecated.variable_scope("roi"):
            image_ids = flow.concat(
                flow.detection.extract_piece_slice_id(rpn_proposals), axis=0
            )
            concat_rpn_proposals = flow.concat(rpn_proposals, axis=0)
            x = self.box_feature_extractor(concat_rpn_proposals, image_ids, features)
            box_pred, cls_logits = self.box_predictor(x)
            results = self.build_post_processor(
                cls_logits, box_pred, rpn_proposals, image_size_list
            )

        return results

    def box_feature_extractor(self, proposals, img_ids, features):
        pooler_scales = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        pooler_res = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        sampling_ratio = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        proposals_with_img_ids = flow.concat(
            [flow.expand_dims(flow.cast(img_ids, flow.float), 1), proposals], axis=1
        )
        levels = flow.detection.level_map(proposals)

        level_idx_list = [
            flow.squeeze(
                flow.local_nonzero(levels == flow.constant_scalar(i, flow.int32)),
                axis=[1],
            )
            for i in range(len(pooler_scales))
        ]

        roi_features_list = [
            flow.detection.roi_align(
                feature,
                rois=flow.local_gather(proposals_with_img_ids, level_idx),
                pooled_h=pooler_res,
                pooled_w=pooler_res,
                spatial_scale=scale,
                sampling_ratio=sampling_ratio,
                name="box_roi_align_" + str(i),
            )
            for i, (feature, level_idx, scale) in enumerate(
                zip(features, level_idx_list, pooler_scales), 1
            )
        ]

        roi_features = flow.stack(roi_features_list, axis=0)
        origin_indices = flow.stack(level_idx_list, axis=0)
        roi_features_reorder = flow.local_scatter_nd_update(
            flow.constant_like(roi_features, float(0)),
            flow.expand_dims(origin_indices, axis=1),
            roi_features,
        )

        roi_features_flat = flow.dynamic_reshape(
            roi_features_reorder,
            [-1, reduce(operator.mul, roi_features_reorder.shape[1:], 1)],
        )

        representation_size = self.cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        x = flow.layers.dense(
            inputs=roi_features_flat,
            units=representation_size,
            activation=flow.keras.activations.relu,
            use_bias=True,
            kernel_initializer=flow.kaiming_initializer(
                shape=(representation_size, roi_features_flat.static_shape[1]),
                distribution="random_uniform",
                mode="fan_in",
                nonlinearity="leaky_relu",
                negative_slope=1.0,
            ),
            bias_initializer=flow.constant_initializer(0),
            name="fc6",
        )

        x = flow.layers.dense(
            inputs=x,
            units=representation_size,
            activation=flow.keras.activations.relu,
            use_bias=True,
            kernel_initializer=flow.kaiming_initializer(
                shape=(representation_size, x.static_shape[1]),
                distribution="random_uniform",
                mode="fan_in",
                nonlinearity="leaky_relu",
                negative_slope=1.0,
            ),
            bias_initializer=flow.constant_initializer(0),
            name="fc7",
        )

        return x

    def box_predictor(self, x):
        bbox_regression = flow.layers.dense(
            inputs=x,
            units=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES * 4,
            activation=None,
            use_bias=True,
            kernel_initializer=flow.random_normal_initializer(stddev=0.001),
            bias_initializer=flow.constant_initializer(0),
            name="bbox_pred",
        )
        cls_logits = flow.layers.dense(
            inputs=x,
            units=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            activation=None,
            use_bias=True,
            kernel_initializer=flow.random_normal_initializer(stddev=0.01),
            bias_initializer=flow.constant_initializer(0),
            name="cls_score",
        )

        return bbox_regression, cls_logits

    # cls_logits: [R, 81], R = sum(R'_i)
    # box_regressions: [R, 324], R = sum(R'_i)
    # rpn_proposals_list: list of [R'_i, 4] wrt. images
    # image_size_list: list of [2,] wrt. images
    def build_post_processor(
        self, cls_logits, box_regressions, rpn_proposals_list, image_size_list
    ):
        cls_probs = flow.nn.softmax(cls_logits)
        concat_rpn_proposals = flow.concat(rpn_proposals_list, axis=0)
        concat_proposals = flow.detection.box_decode(
            concat_rpn_proposals,
            box_regressions,
            regression_weights={
                "weight_x": 10.0,
                "weight_y": 10.0,
                "weight_h": 5.0,
                "weight_w": 5.0,
            },
        )
        proposals_list = flow.detection.maskrcnn_split(
            concat_proposals, rpn_proposals_list
        )
        cls_probs_list = flow.detection.maskrcnn_split(cls_probs, rpn_proposals_list)
        assert len(proposals_list) == len(cls_probs_list) == len(image_size_list)

        result_list = []
        for img_idx, (proposals, cls_probs, image_size) in enumerate(
            zip(proposals_list, cls_probs_list, image_size_list)
        ):
            boxes_list = []
            scores_list = []
            labels_list = []
            # Clip to image
            proposals = flow.detection.clip_to_image(proposals, image_size)

            inds_all = cls_probs > flow.constant_scalar(float(0.05), flow.float)
            # Skip j = 0, because it's the background class
            for j in range(1, 81):
                inds = flow.squeeze(
                    flow.local_nonzero(
                        flow.squeeze(
                            flow.slice_v2(
                                inds_all, [{}, {"begin": j, "end": j + 1, "stride": 1}]
                            ),
                            axis=[1],
                        )
                    ),
                    axis=[1],
                )
                # [R, 4]
                boxes_j = flow.slice_v2(
                    flow.local_gather(proposals, inds),
                    [{}, {"begin": j * 4, "end": (j + 1) * 4, "stride": 1}],
                )
                # [R,]
                scores_j = flow.squeeze(
                    flow.slice_v2(
                        flow.local_gather(cls_probs, inds),
                        [{}, {"begin": j, "end": j + 1, "stride": 1}],
                    ),
                    axis=[1],
                )

                # Apply NMS to boxes
                inds_after_nms = flow.squeeze(
                    flow.local_nonzero(
                        flow.detection.nms(
                            boxes_j, nms_iou_threshold=0.5, post_nms_top_n=-1
                        )
                    ),
                    axis=[1],
                )
                boxes_j = flow.local_gather(boxes_j, inds_after_nms)
                scores_j = flow.local_gather(scores_j, inds_after_nms)
                labels_j = flow.cast(flow.constant_like(scores_j, int(j)), flow.int32)

                boxes_list.append(boxes_j)
                scores_list.append(scores_j)
                labels_list.append(labels_j)

            boxes = flow.concat(boxes_list, axis=0)
            scores = flow.concat(scores_list, axis=0)
            labels = flow.concat(labels_list, axis=0)

            result_inds = flow.math.top_k(scores, 100)
            result = (
                flow.local_gather(boxes, result_inds),
                flow.local_gather(scores, result_inds),
                flow.local_gather(labels, result_inds),
                image_size,
            )

            result_list.append(result)

        return result_list
