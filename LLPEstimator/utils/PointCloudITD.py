import copy
import glob
import math
import os
import re

import geopandas as gpd
import ipywidgets
import laspy
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import shapes
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, maximum_filter
from scipy.stats import binned_statistic_2d
from shapely import points
from shapely.geometry import Point, Polygon
from shapely.geometry import shape
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree
from skimage.measure import find_contours
from skimage.measure import label as sklabel
from skimage.measure import regionprops
from skimage.segmentation import watershed
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_las(pointcloudfile, get_attributes=False, useevery=1):
    """
    :param pointcloudfile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return XYZ (default is False)
    :param useevery: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: 3D array of points (x,y,z) of length number of points in input file (or subsampled by 'useevery')
    """

    # Read file
    inFile = laspy.read(pointcloudfile)
    header = inFile.header

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
        for las_field in las_fields[3:]:  # skip the X,Y,Z fields
            # for las_field in las_fields:  # get all fields
            attributes[las_field] = inFile.points[las_field][::useevery]
        return (coords, attributes, header)


def expand_point_cloud_with_circles(point_cloud, radius, num_points=8):
    """
    Replace each point in the point cloud with a circle of points around it.

    Parameters:
        point_cloud: numpy array of shape (N, 3)
        radius: radius of the circle
        num_points: number of points to place around the circle

    Returns:
        expanded_point_cloud: numpy array of shape (N * num_points, 3)
    """
    N = point_cloud.shape[0]
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x_offsets = radius * np.cos(angles)
    y_offsets = radius * np.sin(angles)

    # For each point, replicate the x, y, z num_points times
    x_repeat = np.repeat(point_cloud[:, 0], num_points)
    y_repeat = np.repeat(point_cloud[:, 1], num_points)
    z_repeat = np.repeat(point_cloud[:, 2], num_points)

    # Tile the offsets for all points
    x_offsets_tile = np.tile(x_offsets, N)
    y_offsets_tile = np.tile(y_offsets, N)

    # Compute the new positions
    x_new = x_repeat + x_offsets_tile
    y_new = y_repeat + y_offsets_tile
    z_new = z_repeat  # z remains the same

    expanded_point_cloud = np.column_stack((x_new, y_new, z_new))

    return expanded_point_cloud


def generate_chm(
    point_cloud, grid_size=1.0, sigma=1.0, subcircle_radius=None, num_subcircle_points=8
):
    # Expand point cloud with subcircles if specified
    if subcircle_radius is not None:
        point_cloud = expand_point_cloud_with_circles(
            point_cloud, subcircle_radius, num_subcircle_points
        )

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    # Create grid bins
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)

    # Compute the maximum z in each grid cell (CHM)
    statistic, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z, statistic="max", bins=[x_bins, y_bins]
    )

    # Replace NaNs with zeros in the CHM
    chm = np.nan_to_num(statistic, nan=0.0)

    # Apply Gaussian filter to smooth CHM
    chm = gaussian_filter(chm, sigma=sigma)

    # Calculate center and radius based on grid boundaries
    grid_center_x = (x_edge[0] + x_edge[-1]) / 2
    grid_center_y = (y_edge[0] + y_edge[-1]) / 2
    radius = np.sqrt((x - grid_center_x) ** 2 + (y - grid_center_y) ** 2).max()

    # Create the circular mask centered on the grid center
    xv, yv = np.meshgrid(
        (x_edge[:-1] + x_edge[1:]) / 2, (y_edge[:-1] + y_edge[1:]) / 2, indexing="ij"
    )
    distance_from_center = np.sqrt(
        (xv - grid_center_x) ** 2 + (yv - grid_center_y) ** 2
    )
    circular_mask = distance_from_center <= radius

    # Mask out values outside the circular area
    chm[~circular_mask] = np.nan

    return chm, x_edge, y_edge


def detect_trees(
    chm, x_edge, y_edge, grid_size=1.0, height_bins=None, window_sizes=None
):
    # Initialize an array to store detected maxima
    total_maxima = np.zeros_like(chm, dtype=bool)

    # Iterate over height bins and corresponding window sizes
    for i in range(len(height_bins) - 1):
        height_min = height_bins[i]
        height_max = height_bins[i + 1]
        window_size = window_sizes[i]

        # Create a mask for the current height bin
        height_mask = (chm >= height_min) & (chm < height_max)

        if not np.any(height_mask):
            continue

        # Apply maximum_filter with the corresponding window size
        local_max = maximum_filter(chm, size=window_size, mode="reflect")

        # Detect local maxima within the height bin
        maxima = (chm == local_max) & height_mask & (chm > 0)

        total_maxima = total_maxima | maxima

    labeled, num_objects = label(total_maxima)

    maxima_coords = np.array(
        ndimage.center_of_mass(chm, labeled, range(1, num_objects + 1))
    )

    # Convert maxima coordinates back to x, y positions (CELL CENTERS)
    tree_tops = []
    for coord in maxima_coords:
        xi, yi = coord
        if np.isnan(xi) or np.isnan(yi):
            continue

        x_center = x_edge[0] + (xi + 0.5) * grid_size
        y_center = y_edge[0] + (yi + 0.5) * grid_size
        tree_tops.append([x_center, y_center])

    return tree_tops


def segment_crowns(chm, tree_tops, x_edge, y_edge, grid_size):
    """
    Segments tree crowns from a Canopy Height Model (CHM) using watershed segmentation.

    Parameters:
    - chm: 2D numpy array representing the Canopy Height Model.
    - tree_tops: List of tuples [(x_center, y_center), ...] representing tree top coordinates.
    - x_edge: Tuple (min_x, max_x) representing the spatial extent in the x-direction.
    - y_edge: Tuple (min_y, max_y) representing the spatial extent in the y-direction.
    - grid_size: Float indicating the size of each grid cell.

    Returns:
    - segmentation_polygons: List of dictionaries with 'label' and 'geometry' (Shapely Polygon or MultiPolygon).
    - marker_coords: List of dictionaries with 'label', 'x', and 'y'.
    """
    # Convert tree_tops spatial coordinates to grid indices
    tree_tops_coords = []
    for x_center, y_center in tree_tops:
        xi = int((x_center - x_edge[0]) / grid_size)
        yi = int((y_center - y_edge[0]) / grid_size)
        tree_tops_coords.append([xi, yi])

    # Create a marker image for watershed from tree tops
    markers = np.zeros_like(chm, dtype=int)
    for i, (xi, yi) in enumerate(tree_tops_coords):
        if 0 <= xi < markers.shape[0] and 0 <= yi < markers.shape[1]:
            markers[xi, yi] = i + 1  # Assign a unique label to each tree top

    # Apply watershed segmentation using the CHM as the topography and tree tops as markers
    segmentation = watershed(-chm, markers, mask=~np.isnan(chm))

    # Label the segmentation
    labeled_seg = sklabel(segmentation)

    # Extract marker coordinates in spatial terms
    marker_coords = []
    for region in regionprops(markers):
        # Calculate the centroid in grid indices
        centroid_i, centroid_j = region.centroid
        # Convert grid indices to spatial coordinates
        x_coord = x_edge[0] + centroid_i * grid_size
        y_coord = y_edge[0] + centroid_j * grid_size
        marker_coords.append({"label": region.label, "x": x_coord, "y": y_coord})

    # Extract exact boundaries of each segmented region
    segmentation_polygons = []
    for region in tqdm(regionprops(labeled_seg), desc="Segmentation"):
        label_id = region.label
        # Create a binary mask for the current region
        region_mask = (labeled_seg == label_id).astype(np.uint8)

        # Find contours at a constant value of 0.5
        contours = find_contours(region_mask, 0.5)

        # Convert contour coordinates to spatial coordinates
        spatial_contours = []
        for contour in contours:
            # Note: find_contours returns coordinates as (row, col)
            # Convert them to (x, y) based on grid_size and edges
            # Adjust for the fact that contours are drawn between pixels
            contour[:, 0] = x_edge[0] + contour[:, 0] * grid_size
            contour[:, 1] = y_edge[0] + contour[:, 1] * grid_size
            spatial_contours.append(contour)

        # Create Shapely polygons from contours
        polygons = []
        for contour in spatial_contours:
            if len(contour) < 3:
                continue  # Not enough points to form a polygon
            try:
                poly = Polygon(contour)
                if poly.is_valid:
                    polygons.append(poly)
                else:
                    # Attempt to fix invalid polygons
                    poly = poly.buffer(0)
                    if poly.is_valid:
                        polygons.append(poly)
            except Exception as e:
                print(f"Error creating polygon for label {label_id}: {e}")

        if not polygons:
            continue  # No valid polygons found

        # If multiple polygons are found, create a MultiPolygon
        if len(polygons) == 1:
            segmentation_polygons.append({"label": label_id, "geometry": polygons[0]})
        else:
            # Merge overlapping polygons
            merged = unary_union(polygons)
            segmentation_polygons.append({"label": label_id, "geometry": merged})

    return segmentation_polygons, marker_coords


def segment_crowns_fast(chm, tree_tops, x_edge, y_edge, grid_size):
    """
    Fast crown segmentation with correct GIS-aligned polygonization.

    Key fix:
      - transpose labeled segmentation to raster order (row=y, col=x)
      - use from_origin(west=x_min, north=y_max, xsize=gs, ysize=gs)
    """
    tt = np.asarray(tree_tops, dtype=float)
    if tt.size == 0:
        return [], []

    # Tree-top -> grid indices in the SAME convention as CHM (axis0=x, axis1=y)
    xi = np.floor((tt[:, 0] - x_edge[0]) / grid_size).astype(np.int32)
    yi = np.floor((tt[:, 1] - y_edge[0]) / grid_size).astype(np.int32)

    H, W = chm.shape
    inside = (xi >= 0) & (xi < H) & (yi >= 0) & (yi < W)
    xi, yi = xi[inside], yi[inside]

    if xi.size == 0:
        return [], []

    labels = np.arange(1, xi.size + 1, dtype=np.int32)

    markers = np.zeros((H, W), dtype=np.int32)
    markers[xi, yi] = labels

    segmentation = watershed(-chm, markers, mask=~np.isnan(chm))
    labeled_seg = sklabel(segmentation).astype(np.int32)

    # Marker coords (CELL CENTERS)
    marker_coords = [
        {
            "label": int(lab),
            "x": float(x_edge[0] + (i + 0.5) * grid_size),
            "y": float(y_edge[0] + (j + 0.5) * grid_size),
        }
        for lab, i, j in zip(labels, xi, yi)
    ]

    # --- FIXED POLYGONIZATION ---
    # Convert (x, y) -> (row=y, col=x) and flip north-up
    labeled_rc = np.flipud(labeled_seg.T)
    valid_mask = np.flipud((~np.isnan(chm)).T) & (labeled_rc > 0)

    transform = from_origin(
        west=float(x_edge[0]),
        north=float(y_edge[-1]),
        xsize=grid_size,
        ysize=grid_size,
    )

    segmentation_polygons = []
    for geom, value in shapes(labeled_rc, mask=valid_mask, transform=transform):
        value = int(value)
        if value == 0:
            continue
        segmentation_polygons.append({"label": value, "geometry": shapely_shape(geom)})

    # Merge multipart pieces per label if needed
    if segmentation_polygons:
        by_label = {}
        for item in segmentation_polygons:
            by_label.setdefault(item["label"], []).append(item["geometry"])
        merged_out = []
        for lab, geoms in by_label.items():
            geom = geoms[0] if len(geoms) == 1 else unary_union(geoms)
            merged_out.append({"label": lab, "geometry": geom})
        segmentation_polygons = merged_out

    return segmentation_polygons, marker_coords


def get_tree_labels(coords, crown_segments):
    segment_labels = np.zeros(coords.shape[0], dtype=np.int32)
    points = [Point(x, y) for x, y, z in coords]

    for segment in tqdm(crown_segments, desc="Segment Point Cloud"):
        label = segment["label"]
        polygon = segment["geometry"]

        for i, point in enumerate(points):
            if polygon.contains(point):
                segment_labels[i] = label

    return segment_labels


def get_tree_labels_fast(coords, crown_segments):
    xy = coords[:, :2]

    # Build shapely points in one shot (fast, vectorized)
    pts = points(xy[:, 0], xy[:, 1])

    # Extract polygons and labels
    polys = [seg["geometry"] for seg in crown_segments]
    labels = np.array([seg["label"] for seg in crown_segments], dtype=np.int32)

    # Spatial index to avoid testing every polygon against every point
    tree = STRtree(pts)

    out = np.zeros(len(pts), dtype=np.int32)

    # For each polygon: only test candidate points whose bbox intersects
    for poly, lab in zip(polys, labels):
        cand_idx = tree.query(poly)  # indices of candidate points
        if len(cand_idx) == 0:
            continue

        # Vectorized contains for those candidates
        mask = poly.contains(pts.take(cand_idx))
        out[cand_idx[mask]] = lab

    return out


# def save_labeled_las(output_file, coords, attrs, header, segment_labels):
#     las = laspy.LasData(header)

#     las.x = coords[:, 0]
#     las.y = coords[:, 1]
#     las.z = coords[:, 2]

#     for attr_name, attr_values in attrs.items():
#         las[attr_name] = attr_values


#     las.add_extra_dim(
#         laspy.ExtraBytesParams(
#             name="TreeID", type=np.int32, description="Tree ID Label"
#         )
#     )
#     las.points["TreeID"] = segment_labels
#     las.write(output_file)
def save_labeled_las(output_file, coords, attrs, header, segment_labels):
    # Create output LasData with same header metadata
    las = laspy.LasData(header)

    # IMPORTANT: resize/allocate points to the subset length
    las.points = laspy.ScaleAwarePointRecord.zeros(len(coords), header=header)

    # Assign coordinates
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]

    # Copy existing dimensions
    for attr_name, attr_values in attrs.items():
        las[attr_name] = attr_values

    # Add TreeID if needed
    if "TreeID" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="TreeID", type=np.int32, description="Tree ID Label"
            )
        )

    las["TreeID"] = segment_labels.astype(np.int32, copy=False)
    las.write(output_file)


def write_chm_geotiff(
    chm,
    x_edge,
    y_edge,
    crs,
    out_path,
    grid_size=0.25,
    nodata=-9999.0,
    compress="deflate",
):
    """
    Write CHM array to GeoTIFF in provided CRS.

    Parameters
    ----------
    chm : np.ndarray
        CHM array indexed as (x, y)
    x_edge : np.ndarray
        X bin edges from CHM generation
    y_edge : np.ndarray
        Y bin edges from CHM generation
    crs : rasterio CRS or pyproj CRS
        CRS parsed from LAS header
    out_path : str
        Output GeoTIFF path
    grid_size : float
        CHM resolution
    nodata : float
        NoData value
    compress : str
        GeoTIFF compression (default: deflate)
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert from (x, y) -> raster order (row=y, col=x) AND flip north-up
    chm_rc = np.flipud(chm.T).astype("float32", copy=False)

    # Replace NaNs with nodata
    if np.isnan(chm_rc).any():
        chm_rc = np.where(np.isnan(chm_rc), nodata, chm_rc)

    transform = from_origin(
        west=float(x_edge[0]),
        north=float(y_edge[-1]),
        xsize=grid_size,
        ysize=grid_size,
    )

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=chm_rc.shape[0],
        width=chm_rc.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress=compress,
    ) as dst:
        dst.write(chm_rc, 1)


if __name__ == "__main__":
    # las_files = [
    #     r"D:\MurrayBrent\projects\paper4\data\rmf_tiles\1kmZ174260532302018L_N.laz",
    # ]

    las_files = [
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174240532102018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174240532202018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174240532302018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174240532402018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174240532502018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174250532102018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174250532202018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174250532302018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174250532402018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174250532502018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174260532102018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174260532202018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174260532302018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174260532402018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174260532502018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174270532102018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174270532202018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174270532302018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174270532402018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174270532502018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174280532102018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174280532202018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174280532302018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174280532402018L_N.laz",
        "D:\\MurrayBrent\\projects\\paper4\\data\\rmf_tiles\\1kmZ174280532502018L_N.laz",
    ]

    for file in tqdm(las_files, desc="Total: "):
        base = os.path.splitext(os.path.basename(file))[0]
        print(f"Reading - {os.path.basename(file)}")
        coords, attrs, header = read_las(file, get_attributes=True)
        crs = header.parse_crs()

        print(f"Generating CHM")
        chm, x_edge, y_edge = generate_chm(
            coords, grid_size=0.25, sigma=1, subcircle_radius=0.3
        )

        chm_path = os.path.join(
            r"D:\MurrayBrent\projects\paper4\data\rmf_chm",
            base + "_chm.tif",
        )

        write_chm_geotiff(chm, x_edge, y_edge, crs, chm_path, grid_size=0.25)

        height_bins = [0, 5, 10, 15, 20, 25, 30, 35, np.inf]
        window_sizes = [7, 9, 11, 13, 15, 17, 19, 21]

        print("Detecting Tree Tops")
        tree_tops = detect_trees(
            chm,
            x_edge,
            y_edge,
            grid_size=0.25,
            height_bins=height_bins,
            window_sizes=window_sizes,
        )
        #         ttops_file = os.path.join(
        #             r"D:\MurrayBrent\projects\paper4\data\rmf_tiles_itd\ttops",
        #             os.path.basename(file)[:-4] + ".gpkg",
        #         )
        #         tree_data = []
        #         for point in tree_tops:
        #             x, y = point
        #             # Convert the spatial coordinates back to array indices (assuming the grid is aligned with x_edge and y_edge)
        #             # This uses rounding; you might adjust this if you need sub-pixel precision.
        #             ix = int(round((x - x_edge[0]) / 0.25))
        #             iy = int(round((y - y_edge[0]) / 0.25))

        #             # Extract the height from the CHM at the nearest pixel
        #             tree_height = chm[ix, iy]

        #             # Create a Shapely Point geometry for the tree top location
        #             pt_geom = Point(x, y)

        #             # Append the record (adjust the dictionary keys as needed)
        #             tree_data.append({"geometry": pt_geom, "height": tree_height})
        #         gdf = gpd.GeoDataFrame(tree_data, crs=crs)
        #         gdf.to_file(ttops_file, driver="GPKG")

        print("Segmenting Crowns")
        crown_segments, markers = segment_crowns_fast(
            chm, tree_tops, x_edge, y_edge, grid_size=0.25
        )
        segment_labels = get_tree_labels_fast(coords, crown_segments)

        print("Writing ITC LAZ Files")
        out_dir = r"D:\MurrayBrent\projects\paper4\data\rmf_tiles_itd\laz"
        os.makedirs(out_dir, exist_ok=True)

        labels = np.asarray(segment_labels)

        # adjust this if your background/unlabeled value is different (often 0 or -1)
        tree_mask = labels > 0
        tree_ids = np.unique(labels[tree_mask])

        base = os.path.splitext(os.path.basename(file))[0]

        for tid in tqdm(tree_ids, desc="Writing tree LAZ files"):
            idx = labels == tid

            coords_i = coords[idx]
            attrs_i = {k: v[idx] for k, v in attrs.items()}
            labels_i = labels[idx]

            out_path = os.path.join(out_dir, f"{base}_tree_{int(tid):05d}.laz")

            # IMPORTANT: don't reuse the same header object; copy it for each subset
            header_i = copy.deepcopy(header)
            save_labeled_las(out_path, coords_i, attrs_i, header_i, labels_i)
