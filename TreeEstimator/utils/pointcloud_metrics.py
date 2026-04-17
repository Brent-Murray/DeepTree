import os

import laspy
import numpy as np
from scipy.spatial import ConvexHull


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)

    # Get coordinates (XYZ)
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    coords = coords[::useevery, :]

    # Return coordinates only
    if get_attributes == False:
        return coords

    # Return coordinates and attributes
    else:
        las_fields = [info.name for info in inFile.points.point_format.dimensions]
        attributes = {}
        # for las_field in las_fields[3:]:  # skip the X,Y,Z fields
        for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return attributes


def calculate_metrics(point_cloud):
    """
    Calculate LiDAR metrics for forest point cloud analysis.

    Metrics computed:
      - Height percentiles: 25th, 50th, 75th, 90th, 95th, 99th.
      - Mean height, max height, and standard deviation of heights.
      - Skewness and kurtosis of height values.
      - Entropy of the height distribution (Shannon diversity index).
      - Signal attenuation depths at the 25th, 50th, and 75th percentiles
        (defined as max height minus the respective percentile).
      - Number of points.
      - Projected area (area of the 2D convex hull of XY coordinates).
      - Convex hull crown volume (volume of the 3D convex hull of points).
      - Rumple index (ratio of 3D convex hull surface area to projected area).
      - Intensity metrics: mean, max, standard deviation, and percentiles
        (25th, 50th, 75th, 90th, 95th, and 99th).

    Parameters:
      point_cloud (dict): Must contain keys 'X', 'Y', 'Z', and 'intensity'.

    Returns:
      dict: Computed LiDAR metrics.
    """
    if len(point_cloud["Z"]) == 0:
        return {}

    metrics = {}

    # Height metrics
    heights = np.array(point_cloud["Z"])
    # metrics["mean_height"] = np.mean(heights)
    # metrics["max_height"] = np.max(heights)
    # metrics["std_dev_height"] = np.std(heights)
    metrics["num_points"] = len(heights)

    # Height percentiles
    # for p in [25, 50, 75, 90, 95, 99]:
    for p in [25]:
        metrics[f"height_p{p}"] = np.percentile(heights, p)

    # Skewness and kurtosis (moment-based)
    # mean_h = metrics["mean_height"]
    # std_h = metrics["std_dev_height"]
    std_h = np.std(heights)
    mean_h = np.mean(heights)
    if std_h > 0:
        metrics["skewness"] = np.mean(((heights - mean_h) / std_h) ** 3)
        metrics["kurtosis"] = np.mean(((heights - mean_h) / std_h) ** 4) - 3
    else:
        metrics["skewness"] = 0
        metrics["kurtosis"] = 0

    # Entropy of height distribution (using 10 bins)
    hist, _ = np.histogram(heights, bins=10)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # remove zeros for log computation
    metrics["entropy_height"] = -np.sum(prob * np.log(prob))

    # Signal attenuation depths (max height minus the percentile heights)
    for p in [25, 50, 75]:
        metrics[f"attenuation_depth_p{p}"] = np.max(heights) - np.percentile(heights, p)

    # Spatial metrics using convex hulls
    X = np.array(point_cloud["X"])
    Y = np.array(point_cloud["Y"])
    pts_2d = np.column_stack((X, Y))
    pts_3d = np.column_stack((X, Y, heights))
    try:
        hull2d = ConvexHull(pts_2d)
        # For 2D hulls, 'volume' is the area
        metrics["projected_area"] = hull2d.volume
    except Exception:
        metrics["projected_area"] = np.nan
    try:
        hull3d = ConvexHull(pts_3d)
        metrics["crown_volume"] = hull3d.volume
        canopy_surface_area = hull3d.area
    except Exception:
        metrics["crown_volume"] = np.nan
        canopy_surface_area = np.nan

    # Rumple index: ratio of canopy surface area to projected area
    if metrics["projected_area"] and metrics["projected_area"] > 0:
        metrics["rumple_index"] = canopy_surface_area / metrics["projected_area"]
    else:
        metrics["rumple_index"] = np.nan

    # Intensity metrics
    intensities = np.array(point_cloud["intensity"])
    # metrics["mean_intensity"] = np.mean(intensities)
    metrics["max_intensity"] = np.max(intensities)
    metrics["std_dev_intensity"] = np.std(intensities)
    # for p in [25, 50, 75, 90, 95, 99]:
    for p in [50]:
        metrics[f"intensity_p{p}"] = np.percentile(intensities, p)

    # Apex angles: compute angles from the apex (highest point) to all other points.
    if len(heights) > 1:
        apex_index = np.argmax(heights)
        apex_x, apex_y, apex_z = X[apex_index], Y[apex_index], heights[apex_index]
        mask = np.arange(len(heights)) != apex_index
        h_diff = apex_z - heights[mask]
        horizontal_dist = np.sqrt((X[mask] - apex_x) ** 2 + (Y[mask] - apex_y) ** 2)
        # Compute angle in degrees: arctan(horizontal_dist / vertical difference)
        angles = np.degrees(np.arctan2(horizontal_dist, h_diff))
        metrics["mean_apex_angle"] = np.mean(angles)
        metrics["std_dev_apex_angle"] = np.std(angles)
    else:
        metrics["mean_apex_angle"] = np.nan
        metrics["std_dev_apex_angle"] = np.nan

    return metrics


# def calculate_metrics(
#     point_cloud, height=True, intensity=True, return_metrics=True, spatial=True
# ):
#     """
#     Calculate a suite of LiDAR metrics for forest point cloud analysis.

#     Parameters:
#         point_cloud (dict): Point cloud data containing fields for height (Z), intensity,
#                             return_number, and number_of_returns for each point.
#         height (bool): Whether to compute height-based metrics.
#         intensity (bool): Whether to compute intensity-based metrics.
#         return_metrics (bool): Whether to compute return number metrics.
#         spatial (bool): Whether to compute spatial metrics.

#     Returns:
#         dict: Selected LiDAR metrics.
#     """
#     # Check if point cloud has any points
#     if len(point_cloud["Z"]) == 0:
#         return {}

#     metrics = {}

#     # Height-based Metrics
#     if height:
#         heights = point_cloud["Z"]
#         height_metrics = {
#             "mean_height": np.mean(heights),
#             "std_dev_height": np.std(heights),
#             "max_height": np.max(heights),
#             "quadratic_mean_height": np.sqrt(np.mean(heights**2)),
#             **{
#                 f"height_p{p}": np.percentile(heights, p)
#                 for p in [25, 50, 75, 90, 95, 99]
#             },
#             "canopy_cover": np.sum(heights > 2.0) / len(heights),
#             "kurtosis": np.mean(((heights - np.mean(heights)) / np.std(heights)) ** 4)
#             - 3,
#             "skewness": np.mean(((heights - np.mean(heights)) / np.std(heights)) ** 3),
#         }
#         # Vertical distribution ratios
#         height_breaks = [2, 5, 10]
#         for i in range(len(height_breaks) - 1):
#             lower, upper = height_breaks[i], height_breaks[i + 1]
#             height_metrics[f"ratio_{lower}_{upper}"] = np.sum(
#                 (heights > lower) & (heights <= upper)
#             ) / len(heights)

#         metrics.update(height_metrics)

#     # Intensity-based Metrics
#     if intensity:
#         intensities = point_cloud["intensity"]
#         intensity_metrics = {
#             "mean_intensity": np.mean(intensities),
#             "std_dev_intensity": np.std(intensities),
#             "max_intensity": np.max(intensities),
#             "intensity_p95": np.percentile(intensities, 95),
#         }
#         metrics.update(intensity_metrics)

#     # Return Number Metrics
#     if return_metrics:
#         return_numbers = point_cloud["return_number"]
#         total_returns = point_cloud["number_of_returns"]
#         return_data = {
#             "mean_return_number": np.mean(return_numbers),
#             "proportion_first_returns": np.sum(return_numbers == 1)
#             / len(return_numbers),
#             "proportion_last_returns": np.sum(return_numbers == total_returns)
#             / len(return_numbers),
#         }
#         metrics.update(return_data)

#     # Spatial Metrics
#     if spatial:
#         x_coords = point_cloud["X"]
#         y_coords = point_cloud["Y"]
#         xy_distances = np.sqrt(
#             (x_coords - np.mean(x_coords)) ** 2 + (y_coords - np.mean(y_coords)) ** 2
#         )
#         spatial_metrics = {
#             "mean_xy_distance": np.mean(xy_distances),
#             "std_dev_xy_distance": np.std(xy_distances),
#             "point_density": len(point_cloud) / (np.ptp(x_coords) * np.ptp(y_coords)),
#         }
#         metrics.update(spatial_metrics)

#     return metrics
