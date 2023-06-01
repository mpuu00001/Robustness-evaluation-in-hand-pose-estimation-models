"""
Author: Pu Muxin
Date: 29/11/2022
"""

import os
import torch
from bodyhands import add_bodyhands_config
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog


class CustomPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs], height, width)[0]
            return predictions


def prepare_model(cfg_file, weights, thresh):
    cfg = get_cfg()
    add_bodyhands_config(cfg)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    cfg.MODEL.WEIGHTS = os.path.abspath(weights)
    predictor = CustomPredictor(cfg)
    return predictor


def detect_hand_rectangle(image):
    model = prepare_model('bodyhands/configs/BodyHands.yaml', 'bodyhands/models/model.pth', 0.7)
    outputs = model(image)
    outputs = outputs["instances"].to("cpu")
    classes = outputs.pred_classes
    boxes = outputs.pred_boxes.tensor
    hand_indices = classes == 0
    hand_boxes = boxes[hand_indices]
    hand_boxes_copy = [[float(hand_boxes[i][j]) for j in range(len(hand_boxes[i]))] for i in range(len(hand_boxes))]
    return hand_boxes_copy
