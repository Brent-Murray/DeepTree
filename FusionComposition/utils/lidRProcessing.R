# List of packages to load/install
packages <- c("lidR","future", "parallel", "sf")

# Install Packages if not Installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)){
  install.packages(packages[!installed_packages])
}

# Load Packages
invisible(suppressMessages(suppressWarnings(lapply(packages, library, character.only = TRUE))))

# Remove Variables
rm(installed_packages)
rm(packages)


# Read in lidr Functions Script
source("D:/MurrayBrent/scripts/old_scripts/R/lidrFunctions.R")

# Set parallelization
plan(multisession, workers = 15L)

ctg <- readLAScatalog("E:/RMF_SPL100/LAS_Classified_Point_Clouds")
opt_stop_early(ctg) <- FALSE # dont stop early
dem <- grid_terrain(ctg, 20, tin())

for (i in 1:length(dem)){
  raster <- dem[[i]]
  x <- terra::xmin(raster)
  y <- terra::ymax(raster)
  
  output_name = paste0("dem_", x, "_", y, ".tif")
  output_path = file.path("E:/RMF_SPL100/dem", output_name)
  terra::writeRaster(raster, output_path, format="GTiff")
}



terra::writeRaster(dem, "E:/RMF_SPL100/dem/dem.tif")


# Create Plots
# Read in lidar
# Set parallelization
plan(multisession, workers = 15L)
ctg <- readLAScatalog("E:/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized", filter = '-drop_z_below 2')
output <- "E:/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized_Retiled"

opt_chunk_buffer(ctg) <- 50
opt_chunk_size(ctg) <- 1000
opt_laz_compression(ctg) <- TRUE
opt_output_files(ctg) <- paste0(output, "/{XLEFT}_{YBOTTOM}")
new_ctg = catalog_retile(ctg)

ctg <- readLAScatalog("E:/RMF_SPL100/LAS_Classified_Point_Clouds_Normalized_Retiled")
plots <- st_read("D:/MurrayBrent/projects/paper2/data/raw/RMF_SPL/RMF_plots/pixel_center.shp")
footprint <- as.spatial(ctg)
footprint <- st_as_sf(footprint)
footprint <- st_transform(footprint, crs(plots))
st_write(footprint, "E:/RMF_SPL100/Retiled_Index/footprint.shp")

plots_tiles <- st_join(plots, footprint[, "filename"])
#### TOO SLOW
# plots <- st_read("D:/MurrayBrent/projects/paper2/data/raw/RMF_SPL/RMF_plots/plots.gpkg")
# 
# # Split Dataset into chunks
# # Needed to split to actually run clip_roi
# chunks <- split(plots, cut(1:nrow(plots), breaks = 764, labels=FALSE))
# 
# # Remove plots from environment
# rm(plots)
# gc()
# 
# # Output Folder
# output <- "F:/paper2/RMF_LAZ_Plots"
# 
# # Create Output folder if it doesnt exist
# if(!dir.exists(output)){
#   dir.create(output)
# }
# 
# # clip pointclouds
# for (i in 1:length(chunks)){
#   out_folder <- paste0(output, "/chunk_", i)
#   if(!dir.exists(out_folder)){
#     dir.create(out_folder)
#   }
#   opt_stop_early(ctg) <- FALSE
#   opt_output_files(ctg) <- paste0(out_folder, "/{id}")
#   opt_laz_compression(ctg) <- TRUE
#   print(paste0("Clipping: ", i,"/",length(chunks)))
#   chunk_ctg <- clip_roi(ctg, chunks[[i]])
# }
