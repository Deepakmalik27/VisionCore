"""#23: ask the detector only for the classes we use.

The p0verify run showed 'person/head split' dropping 65.5% of detections and
the funnel warning that no downstream count could be trusted. It was COCO
furniture -- chairs, potted plants, vases -- being detected and discarded.
"""
from kevacv.engine import _det_classes, _detector_has_head_class


class _M:
    def __init__(self, names):
        self.names = names


def test_stock_coco_asks_for_person_only():
    assert _det_classes(False) == [0]


def test_two_class_finetune_keeps_the_head_class():
    assert _det_classes(True) == [0, 1]


def test_coco_bicycle_is_never_mistaken_for_a_head():
    coco = _M({0: "person", 1: "bicycle", 2: "car"})
    assert _detector_has_head_class(coco) is False
    assert _det_classes(_detector_has_head_class(coco)) == [0]


def test_real_person_head_model_is_recognised():
    m = _M({0: "person", 1: "head"})
    assert _detector_has_head_class(m) is True
    assert _det_classes(_detector_has_head_class(m)) == [0, 1]
