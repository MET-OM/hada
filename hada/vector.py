import logging
import numpy as np
import pyproj

logger = logging.getLogger(__name__)

def rotate_vectors(reader_x, reader_y, u_component, v_component,
                    proj_from, proj_to):
    """Rotate vectors from one crs to another."""

    # if type(proj_from) is str:
    #     proj_from = pyproj.Proj(proj_from)
    # if type(proj_from) is not pyproj.Proj:
    #     proj_from = pyproj.Proj('+proj=latlong +R=6370997.0 +ellps=WGS84')
    #     reader_x, reader_y = self.xy2lonlat(reader_x, reader_y)
    # if type(proj_to) is str:
    #     proj_to = pyproj.Proj(proj_to)

    if proj_from.crs.is_geographic:
        delta_y = .1  # 0.1 degree northwards
    else:
        delta_y = 10  # 10 m along y-axis

    transformer = pyproj.Transformer.from_proj(proj_from, proj_to)
    x2, y2 = transformer.transform(reader_x, reader_y)
    x2_delta, y2_delta = transformer.transform(reader_x,
                                                reader_y + delta_y)

    if proj_to.crs.is_geographic:
        geod = pyproj.Geod(ellps='WGS84')
        rot_angle_vectors_rad = np.radians(
            geod.inv(x2, y2, x2_delta, y2_delta)[0])
    else:
        rot_angle_vectors_rad = np.arctan2(x2_delta - x2, y2_delta - y2)
    logger.debug('Rotating vectors between %s and %s degrees.' %
                    (np.degrees(rot_angle_vectors_rad).min(),
                    np.degrees(rot_angle_vectors_rad).max()))
    rot_angle_rad = -rot_angle_vectors_rad
    u_rot = (u_component * np.cos(rot_angle_rad) -
                v_component * np.sin(rot_angle_rad))
    v_rot = (u_component * np.sin(rot_angle_rad) +
                v_component * np.cos(rot_angle_rad))

    return u_rot, v_rot
