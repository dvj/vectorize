import json
import logging
import math

import cv2
import numpy as np
try:
    from skimage import measure
except ImportError:
    measure = None

logger = logging.getLogger(__name__)


def compute_contours(data, threshold_value, transform=None, exact=False, smooth_value=0,
                     contour_method=cv2.CHAIN_APPROX_TC89_L1, zoom=None):
    if exact:
        thresh_img = cv2.compare(data, threshold_value, cv2.CMP_EQ)
    else:
        ret, thresh_img = cv2.threshold(data, threshold_value, 255, 0)
    if thresh_img.dtype != np.uint8:
        thresh_img = thresh_img.astype(np.uint8)

    if zoom:
        thresh_img = cv2.resize(thresh_img, (thresh_img.shape[0]*zoom, thresh_img.shape[1]*zoom),
                                interpolation=cv2.INTER_NEAREST)
        thresh_img = cv2.GaussianBlur(thresh_img, (3, 3), 0)
        ret, thresh_img = cv2.threshold(thresh_img, 99, 255, 0)
    else:
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        thresh_img = cv2.dilate(thresh_img, cross)
    if thresh_img.dtype != np.uint8:
        thresh_img = thresh_img.astype(np.uint8)
    if contour_method.lower() in ['simple', 'approx']:
        if contour_method.lower == 'simple':
            method = cv2.CHAIN_APPROX_SIMPLE
        else:
            method = cv2.CHAIN_APPROX_TC89_L1
        thresh_img, contours, order = cv2.findContours(thresh_img, cv2.RETR_LIST, method)
    elif contour_method.lower == 'accurate':
        try:
            contours = measure.find_contours(thresh_img, threshold_value, fully_connected='high')
        except AttributeError as e:
            logger.error("Need package 'skimage' for accurate contour method")
            raise e
    else:
        logger.error("Unknown contour method: \"{}\"".format(contour_method))
        contours = []
    need_column_swap = False
    del thresh_img
    featurelist = []
    contour_len = [len(contour) for contour in contours]
    for idx, c_len in enumerate(contour_len):
        if c_len > 1:
            if smooth_value > 0:
                contour = cv2.approxPolyDP(contours[idx].astype(np.float32), smooth_value, True)
            else:
                contour = contours[idx]
            l = len(contour)
            if l < 2:
                continue
            contour = np.reshape(contour, [l, 2])
            if transform is not None:
                # turn into homogeneous coordinates
                con_h = np.ones([l, 3])
                # Doing the below is much quicker than the normal:
                #    contour = np.append(contour, np.ones([l, 1]), axis=1)
                if need_column_swap:
                    con_h[:, :-1] = contour[:, [1, 0]]
                else:
                    con_h[:, :-1] = contour
                # apply transform from pixel coords into srs
                contour = np.dot(con_h, transform).tolist()
            else:
                if need_column_swap:
                    contour[:, [0, 1]] = contour[:, [1, 0]]
                contour = contour.tolist()
            if contour[0] != contour[-1]:
                contour.append(contour[0])
            poly = contour
            featurelist.append([poly])
    return featurelist


def vectorize(data, transform, thresholds=(), threshold_step=None, exact=False,
              out_var_name='data', geometry='polygon'):
    if (not thresholds and not threshold_step) or (thresholds and threshold_step):
        logger.error("Must specify 'thresholds' or 'threshold_step' (exclusively)")
        raise Exception("Invalid arguments")

    smooth_value = 0
    contour_method = 'simple'

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(data)
    logger.debug("Min: {}  Max: {}".format(min_val, max_val))
    if threshold_step:
        thresholds = range(int(math.floor(min_val)), int(math.ceil(max_val)), threshold_step)
    if min_val < min(thresholds):
        logger.warn("Value {} exists below minimum threshold {}".format(min_val, min(thresholds)))
    if max_val > max(thresholds):
        logger.warn("Value {} exists above maximum threshold {}".format(max_val, max(thresholds)))

    # get data into appropriate type:
    if data.dtype in [np.int64, np.uint64]:
        data = data.astype(np.uint8)

    output_data = list()
    for thresh_value in thresholds:
        logger.debug("Processing {}".format(thresh_value))
        contours = compute_contours(data, thresh_value, transform, exact,
                                    smooth_value, contour_method)
        if contours:
            properties = dict({out_var_name: thresh_value})
            if isinstance(thresholds, dict):
                properties.update(thresholds[thresh_value])
            if geometry.lower() == 'polygon':
                geo_data = [dict({'type': 'Feature',
                                  'geometry': {"type": "MultiPolygon",
                                               "coordinates": contours},
                                  'properties': properties
                                  })]
            elif geometry.lower() == 'linestring':
                geo_data = list()
                for c in contours:
                    geo_data.append(dict({'type': 'Feature',
                                          'geometry': {"type": "MultiLineString",
                                                       "coordinates": c},
                                          'properties': properties
                                          }))
            else:
                logger.error("Unknown geometry type: {}".format(geometry))
                geo_data = []
            output_data.append(geo_data)
    geojson_data = {"type": "FeatureCollection", "features": output_data}
    return geojson_data


def write_geojson_file(geojson_data, filename):
    with open("{}".format(filename), 'w') as output_file:
        output_file.write(json.dumps(geojson_data))
