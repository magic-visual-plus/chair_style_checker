
import os
import torch

from chair_style_checker.common import (
    parse_style_configs, search_style_configs, parse_way_point_id,
    StyleCheckResult,
    StyleCheckRecord,
)
import cv2
import loguru
import numpy as np
import tempfile
from collections import defaultdict


logger = loguru.logger



class StylePredictorNetwork:
    def __init__(self, model_path: str, api_key: str, model_id: str):
        self.internal_predictor = StylePredictorInternal(model_path)
        from model_sdk import ModelManager, InferenceModel
        model_mng = ModelManager(api_url="http://127.0.0.1:18900", api_key=api_key)
        self.det_model: InferenceModel = model_mng.model(model_id=model_id)
        pass

    def predict(self, sn, way_point, img_path):
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (512, 512))
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpname = os.path.join(tmpdir, "temp.jpg")
            cv2.imencode('.jpg', img)[1].tofile(tmpname)
            result = self.det_model.predict(request_id="tempid", source=tmpname, encode=False, confidence=0.5)
            pass
        
        names = []
        boxes = []
        for pred in result.predictions:
            names.append(pred.name)
            boxes.append([pred.points[0].x, pred.points[0].y, pred.points[0].x + pred.points[0].w, pred.points[0].y + pred.points[0].h])
            pass

        return self.internal_predictor.predict(
            sn, way_point, img, boxes, names)
        pass


class StylePredictorInternal:
    def __init__(self, model_path: str, api_key: str = None, model_id: str = None):
        config_path = os.path.join(model_path, "style.json")
        self.style_configs = parse_style_configs(config_path)
        self.load_det_model(model_path, api_key, model_id)
        pass

    def load_det_model(self, model_path: str, api_key: str, model_id: str):
        from model_sdk import ModelManager, InferenceModel
        model_mng = ModelManager(api_url="http://127.0.0.1:18900", api_key=api_key)
        self.det_model: InferenceModel = model_mng.model(model_id=model_id)

    def predict_det_model(self, img):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpname = os.path.join(tmpdir, "temp.jpg")
            cv2.imencode('.jpg', img)[1].tofile(tmpname)
            result = self.det_model.predict(request_id="tempid", source=tmpname, encode=False, confidence=0.5)
            pass
        
        names = []
        boxes = []
        for pred in result.predictions:
            names.append(pred.name)
            boxes.append([pred.points[0].x, pred.points[0].y, pred.points[0].x + pred.points[0].w, pred.points[0].y + pred.points[0].h])
            pass
        return boxes, names
    
    def predict(self, product_type, way_point, img) -> StyleCheckResult:

        if way_point is None:
            return StyleCheckResult()
        
        way_point = parse_way_point_id(way_point)
        styles = search_style_configs(way_point, product_type, self.style_configs)
        if len(styles) == 0:
            # logger.info("no style config found for sn: {}, way_point: {}", sn, way_point)
            return StyleCheckResult()
        
        print(way_point, product_type)

        boxes, names = self.predict_det_model(img)
        print("detected boxes:", boxes, names)
        
        detected_target_styles = defaultdict(list)
        for box, name in zip(boxes, names):
            target = name.rsplit('_', 1)[0]
            style = name.rsplit('_', 1)[1]

            if target in detected_target_styles:
                continue
            detected_target_styles[target].append((style, box))
            pass

        result = StyleCheckResult()
        result.product_type = product_type
        for style in styles:
            target_style = detected_target_styles.get(style.target, None)
            if target_style is None:
                # missing target style
                result.missing_targets.append(style.target)
                pass
            else:
                if style.style not in [s for s, b in target_style]:
                    # wrong style
                    print(f"wrong style for target {style.style}, detected: {target_style}")
                    for s, b in target_style:
                        record = StyleCheckRecord(
                            target=style.target,
                            style=s,
                            bounding_box=[int(v) for v in b]
                        )
                        result.wrong_styles.append(record)
                        pass
                    pass
                else:
                    # correct style
                    for s, b in target_style:
                        if s == style.target:
                            record = StyleCheckRecord(
                                target=style.target,
                                style=s,
                                bounding_box=[int(v) for v in b]
                            )
                            result.correct_styles.append(record)
                            pass
                        pass
                    pass
                pass
            pass

        return result
        pass
    pass


class StylePredictor(StylePredictorInternal):
    def __init__(self, model_path: str, api_key: str = None, model_id: str = None):
        super().__init__(model_path, api_key, model_id)
        self.load_det_model(model_path, api_key, model_id)
        pass

    def load_det_model(self, model_path, api_key, model_id):
        from hq_det.models.rtdetr import HQRTDETR
        self.det_model = HQRTDETR(model=os.path.join(model_path, "det_model.pth"))
        if torch.cuda.is_available():
            self.det_model.to('cuda:0')
            pass
        self.det_model.eval()

    def predict_det_model(self, img):
        res = self.det_model.predict([img], bgr=True, confidence=0.5)[0]
        boxes = res.bboxes.tolist()
        names = [self.det_model.id2names[c] for c in res.cls.tolist()]
        return boxes, names
    pass