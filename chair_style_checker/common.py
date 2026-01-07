import pydantic
from typing import List
import json

class Style(pydantic.BaseModel):
    target: str
    style: str

    class Config:
        extra = pydantic.Extra.forbid

class ProductType(pydantic.BaseModel):
    name: List[str]

class StyleConfig(pydantic.BaseModel):
    product_type: ProductType
    way_point: str
    styles: List[Style]

    class Config:
        extra = pydantic.Extra.forbid
        pass
    pass

class StyleCheckRecord(pydantic.BaseModel):
    target: str
    style: str
    bounding_box: List[int]
    pass

class StyleCheckResult(pydantic.BaseModel):
    product_type: str = None
    missing_targets: List[str] = []
    wrong_styles: List[StyleCheckRecord] = []
    correct_styles: List[StyleCheckRecord] = []
    pass


def parse_style_configs(filename) -> List[StyleConfig]:
    with open(filename, 'r', encoding='utf-8') as f:
        config_json = json.load(f)
        pass

    style_configs = []
    for item in config_json:
        style_config = StyleConfig(**item)
        style_configs.append(style_config)
        pass
    return style_configs
    pass

def search_style_configs(way_point, product_type, style_configs) -> List[Style]:
    styles = []
    for style_config in style_configs:
        if way_point != style_config.way_point:
            continue
        t = product_type
        if t not in style_config.product_type.name:
            continue
        styles.extend(style_config.styles)
        pass

    return styles
    pass

def search_style_configs_by_sn(way_point, sn, style_configs: List[StyleConfig]) -> List[Style]:
    styles = []
    product_type = None
    for style_config in style_configs:
        if way_point != style_config.way_point:
            continue
        t = sn[style_config.product_type.start_index:style_config.product_type.end_index]
        if t not in style_config.product_type.name:
            continue
        product_type = t
        styles.extend(style_config.styles)
        pass

    return styles, product_type


def parse_way_point_id(way_point_id: str) -> str:
    if way_point_id is None:
        return "wrong_way_point"
    
    if way_point_id.startswith('_p_'):
        return way_point_id[1:6]
    elif way_point_id.startswith('backend_'):
        return way_point_id
    else:
        return way_point_id
    pass