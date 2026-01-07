import time
import cv2
import os
import sys
import json
import pymysql
pymysql.install_as_MySQLdb()

import sqlalchemy
from sqlalchemy import text
from typing import List
from tqdm import tqdm
import numpy as np
from chair_style_checker.common import StyleConfig, parse_style_configs
from chair_style_checker import predictor
import shutil


if __name__ == '__main__':
    model_path = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    output_path = sys.argv[4]

    style_predictor = predictor.StylePredictorInternal(
        model_path = model_path, api_key="584ff3a5-2bff-4e5f-98e7-7af7e7600cf1", model_id="53807208-8b12-4380-b3f9-cee6f64df832")

    sql_url = 'mysql+mysqldb://root:12345678@127.0.0.1:3306/vision_backend'

    engine = sqlalchemy.create_engine(sql_url, poolclass=sqlalchemy.pool.NullPool)
    connection = engine.connect()

    result = connection.execute(text((
        f'select local_pic_url,way_point_id,product_type from product_detection_detail_result where'
        f' c_time >= "{start_date}" and c_time <= "{end_date}"'
        # f' and local_pic_url like "%a9404c3cab94409aba6a47be94f2e484_d7d58de58beb4237ad%"'
    )))

    cnt = 0
    for row in tqdm(result):
        image_path = row[0]
        product_type = row[2]
        # get way point
        basename = os.path.basename(image_path).strip('_')
        basename = os.path.splitext(basename)[0]
        way_point = row[1]

        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        result = style_predictor.predict(
            product_type, way_point, img
        )

        if len(result.missing_targets) > 0 or len(result.wrong_styles) > 0:
            print(way_point, product_type, result)
            shutil.copy(image_path, os.path.join(
                output_path, product_type + '_' + os.path.basename(image_path)))
            pass
        pass
    
    connection.close()
    engine.dispose()
    pass