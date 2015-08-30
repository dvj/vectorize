import gdal
import logging
import numpy as np
import sys
import vectorize

logger = logging.getLogger(__name__)


def read_image(filename, band=1, no_data_value=None):
    gdal.UseExceptions()
    try:
        ds = gdal.Open(filename)
    except RuntimeError as e:
        logger.exception("Unable to open {}".format(filename))
        raise e
    logger.info("Reading {} file of size: {}x{}x{}".format(ds.GetDriver().ShortName,
                                                           ds.RasterXSize, ds.RasterYSize,
                                                           ds.RasterCount))
    transform = ds.GetGeoTransform()

    try:
        data_band = ds.GetRasterBand(band)
    except RuntimeError as e:
        logger.exception("Could not get band {}".format(band))
        logger.info("Bands available in {}: {}".format(filename, ds.RasterCount))
        raise e

    data = data_band.ReadAsArray()

    # replace the no data value entries with a user supplied value, if applicable
    if no_data_value is not None and data_band.GetNoDataValue():
        if gdal.GetDataTypeName(data_band.DataType).startswith("Float"):
            masked_data = np.ma.masked_values(data, data_band.GetNoDataValue(), copy=False)
        else:
            masked_data = np.ma.masked_equal(data, data_band.GetNoDataValue(), copy=False)
        masked_data.fill_value = no_data_value
        masked_data = np.ma.fix_invalid(masked_data, copy=False)
        data = masked_data.data

    return data, transform


def define_array():
    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 2, 2, 3, 3, 2, 1, 0],
                  [0, 1, 2, 2, 3, 4, 4, 4, 1, 0],
                  [0, 1, 2, 3, 4, 4, 5, 1, 0, 0],
                  [0, 1, 3, 4, 3, 4, 5, 1, 0, 0],
                  [0, 1, 3, 4, 3, 4, 5, 1, 0, 0],
                  [0, 1, 2, 3, 4, 4, 5, 1, 0, 0],
                  [0, 1, 2, 2, 3, 4, 4, 4, 1, 0],
                  [0, 1, 1, 2, 2, 3, 3, 2, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                  ])
    return a

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    d, t = read_image(sys.argv[1], no_data_value=0)
    a_transform = np.array([[t[1], t[2]], [t[4], t[5]], [t[0], t[3]]])
    vectorize.vectorize(d, a_transform, range(0, 96, 1), exact=True)
