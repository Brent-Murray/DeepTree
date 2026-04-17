# TreeEstimator
This is the implementation of the preprocessing and modeling of [Individual tree species prediction using airborne laser scanning data and derived point-cloud metrics within a dual-stream deep learning approach](https://doi.org/10.1016/j.jag.2025.104877).

This paper can be cited as:

Murray, B. A., Coops, N. C., White, J. C., Dick, A., Barbeito, I., & Ragab, A. (2025). Individual tree species prediction using airborne laser scanning data and derived point-cloud metrics within a dual-stream deep learning approach. International Journal of Applied Earth Observation and Geoinformation, 144, 104877. https://doi.org/10.1016/j.jag.2025.104877


Contents
----
```text
├── TreeEstimator/
│   ├── models/
│   │   ├── DGCNN.py
│   │   ├── KPConv.py
│   │   ├── PointExtractor.py
│   │   ├── PointTransformer-Copy1.py
│   │   ├── PointTransformer.py
│   │   ├── SpeciesEstimation-Copy1.py
│   │   ├── SpeciesEstimation.py
│   │   ├── SpeciesEstimationMetrics.py
│   │   ├── SpeciesEstimationPoint.py
│   │   ├── TabNet.py
│   │   └── TreeExtractor.py
│   ├── utils/
│   │   ├── loss_utils.py
│   │   ├── pointcloud_metrics.py
│   │   ├── resample_point_clouds.py
│   │   ├── send_telegram.py
│   │   ├── tools.py
│   │   └── train.py
│   ├── main.py
│   └── README.md
```
