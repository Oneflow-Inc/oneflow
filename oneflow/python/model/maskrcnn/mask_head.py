import oneflow as flow


class MaskHead(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # In mask_head, we reuse pos_proposals and pos_gt_indices from box_head,
    # instead of using matcher to calculate them. We can do this because
    # mask_head and box_head use the same matcher in R_50_FPN_1x training.
    # TODO: add matcher to mask_head for the case that mask_head and box_head
    # use different matchers.

    # pos_proposals: list of [num_pos_rois, 4] wrt. images
    # pos_gt_indices: list of [num_pos_rois,] wrt. images
    # gt_segms: list of (G, 7, 7) wrt. images
    # gt_labels: list of (G,) wrt. images
    # features: list of [N, C_i, H_i, W_i] wrt. fpn layers
    def build_train(
        self, pos_proposals, pos_gt_indices, gt_segms, gt_labels, features
    ):
        with flow.deprecated.variable_scope("mask"):
            img_ids = flow.concat(
                flow.detection.extract_piece_slice_id(pos_proposals), axis=0
            )
            proposals = flow.concat(pos_proposals, axis=0)

            # mask head feature extractor
            x = self.mask_feature_extractor(proposals, img_ids, features)

            # mask head predictor
            mask_fcn_logits = self.mask_predictor(x)

            gt_segm_list = []
            gt_label_list = []
            for img_idx in range(self.cfg.TRAINING_CONF.IMG_PER_GPU):
                gt_segm_list.append(
                    flow.local_gather(
                        gt_segms[img_idx], pos_gt_indices[img_idx]
                    )
                )
                gt_label_list.append(
                    flow.local_gather(
                        gt_labels[img_idx], pos_gt_indices[img_idx]
                    )
                )
            gt_segms = flow.concat(gt_segm_list, axis=0)
            gt_labels = flow.concat(gt_label_list, axis=0)

            mask_pred = flow.keras.activations.sigmoid(
                flow.squeeze(
                    flow.gather(
                        params=mask_fcn_logits,
                        indices=flow.expand_dims(gt_labels, 1),
                        batch_dims=1,
                    ),
                    axis=[1],
                )
            )

            mask_loss = flow.math.reduce_sum(
                flow.nn.sigmoid_cross_entropy_with_logits(gt_segms, mask_pred)
            )
            elem_cnt = flow.elem_cnt(gt_labels, dtype=mask_loss.dtype)
            mask_loss = mask_loss / elem_cnt
            return mask_loss

    def build_eval(self, proposals, features):
        image_ids = flow.detection.extract_piece_slice_id(proposals)
        x = self.mask_feature_extractor(proposals, image_ids, features)
        mask_logits = self.predictor(x)

        return mask_logits

    def mask_feature_extractor(self, proposals, img_ids, features):
        levels = flow.detection.level_map(proposals)
        level_idx_dict = {}
        for (i, scalar) in zip(range(2, 6), range(0, 4)):
            level_idx_dict[i] = flow.local_nonzero(
                levels == flow.constant_scalar(int(scalar), flow.int32)
            )
        proposals_with_img_ids = flow.concat(
            [flow.expand_dims(flow.cast(img_ids, flow.float), 1), proposals],
            axis=1,
        )
        roi_features_list = []
        for (level, i) in zip(range(2, 6), range(0, 4)):
            roi_feature_i = flow.detection.roi_align(
                features[0],
                rois=flow.local_gather(
                    proposals_with_img_ids,
                    flow.squeeze(level_idx_dict[level], axis=[1]),
                ),
                pooled_h=self.cfg.MASK_HEAD.POOLED_H,
                pooled_w=self.cfg.MASK_HEAD.POOLED_W,
                spatial_scale=self.cfg.MASK_HEAD.SPATIAL_SCALE / pow(2, i),
                sampling_ratio=self.cfg.MASK_HEAD.SAMPLING_RATIO,
            )
            roi_features_list.append(roi_feature_i)
        roi_features = flow.stack(roi_features_list, axis=0)
        origin_indices = flow.stack(list(level_idx_dict.values()), axis=0)
        x = flow.local_gather(
            roi_features, flow.squeeze(origin_indices, axis=[1])
        )
        for i in range(1, 5):
            x = flow.layers.conv2d(
                inputs=x,
                filters=256,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                data_format="NCHW",
                dilation_rate=[1, 1],
                activation=flow.keras.activations.relu,
                use_bias=True,
                name="fcn{}".format(i),
            )

        return x

    def mask_predictor(self, x):
        filter = flow.get_variable(
            "conv5-weight",
            shape=(x.static_shape[1], 256, 2, 2),
            dtype=x.dtype,
            initializer=flow.constant_initializer(0),
        )
        bias = flow.get_variable(
            name="conv5-bias",
            shape=(256,),
            dtype=x.dtype,
            initializer=flow.constant_initializer(0),
            model_name="bias",
        )
        x = flow.nn.conv2d_transpose(
            x,
            filter=filter,
            data_format="NCHW",
            padding="same",
            strides=[2, 2],
            name="conv5",
        )
        x = flow.nn.bias_add(x, bias, "NCHW")
        x = flow.keras.activations.relu(x)
        x = flow.layers.conv2d(
            x,
            filters=81,
            kernel_size=[1, 1],
            data_format="NCHW",
            padding="SAME",
            strides=[1, 1],
            dilation_rate=[1, 1],
            name="fcn_logits",
        )

        return x
