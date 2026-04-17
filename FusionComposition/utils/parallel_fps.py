import laspy
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
import open3d as o3d

def farthest_point_sampling(coords, k, attrs=None):
    # Adapted from https://minibatchai.com/sampling/2021/08/07/FPS.html

    # Get points into numpy array
    points = np.array(coords)

    # Get points index values
    idx = np.arange(len(coords))

    # Initialize use_idx
    use_idx = np.zeros(k, dtype="int")

    # Initialize dists
    dists = np.ones_like(idx) * float("inf")

    # Select a point from its index
    selected = 0
    use_idx[0] = idx[selected]

    # Delete Selected
    idx = np.delete(idx, selected)

    # Iteratively select points for a maximum of k samples
    for i in range(1, k):
        # Find distance to last added point and all others
        last_added = use_idx[i - 1]  # get last added point
        dist_to_last_added_point = ((points[last_added] - points[idx]) ** 2).sum(-1)

        # Update dists
        dists[idx] = np.minimum(dist_to_last_added_point, dists[idx])

        # Select point with largest distance
        selected = np.argmax(dists[idx])
        use_idx[i] = idx[selected]

        # Update idx
        idx = np.delete(idx, selected)

    coords = coords[use_idx]
    if attrs:
        for key, vals in attrs.items():
            attrs[key] = vals[use_idx]
        return coords, attrs
    else:
        return coords
    
def write_las(outpoints, outfilepath, attribute_dict={}):
    """
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    """
    import laspy

    hdr = laspy.LasHeader(version="1.4", point_format=6)
    hdr.x_scale = 0.01
    hdr.y_scale = 0.01
    hdr.z_scale = 0.01
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    las = laspy.LasData(hdr)

    las.x = outpoints[:,0]
    las.y = outpoints[:,1]
    las.z = outpoints[:,2]
    for key, vals in attribute_dict.items():
        try:
            las[key] = vals
        except:
            las.add_extra_dim(laspy.ExtraBytesParams(name=key, type=type(vals[0])))
            las[key] = vals

    las.write(outfilepath)
    
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
        return (coords, attributes)
    

# Define a function for processing a single LAS file
def process_las_file(las_file, out_folder):
    if not os.path.exists(os.path.join(out_folder, os.path.basename(las_file))):
        try:
            las = read_las(las_file)
            if len(las) >= 7168:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(las)
                las_o3d = o3d.geometry.PointCloud.farthest_point_down_sample(pc, 7168)
                las_fps = np.asarray(las_o3d.points)
                # las_idx = las[fps_idx]
            else:
                idx = np.random.choice(las.shape[0], 7168, replace=True)
                las_fps = las[idx, :]
            write_las(las_fps, os.path.join(out_folder, os.path.basename(las_file)))
        except Exception as e:
            pass
    else:
        pass

if __name__ == '__main__':
    folder = r"F:\paper2\RMF_LAZ_Plots_circle"
    out_folder = r"F:\paper2\resampled"
    os.makedirs(out_folder, exist_ok=True)
    print("Finding Files")
    las_files = [
        os.path.join(folder, file) 
        for file in os.listdir(folder)
        if file.endswith(".laz")
    ]

    pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 10))
    items_processed = 0

    print("Start Multiprocessing")
    # Define a new callback function that increments and prints the progress
    def update_progress(result):
        global items_processed
        items_processed += 1
        print(f"Processed {items_processed} out of {len(las_files)} files.", end='\r')
        
    for las_file in las_files:
        pool.apply_async(process_las_file, args=(las_file, out_folder), callback=update_progress)
    
#     with tqdm(total=len(las_files)) as pbar:
#         def callback(result):
#             pbar.update(1)

#         for las_file in las_files:
#             pool.apply_async(process_las_file, args=(las_file, out_folder), callback=callback)

    # Close the pool to free up resources
    pool.close()
    pool.join()
