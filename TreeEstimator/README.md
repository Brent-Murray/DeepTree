# TreeEstimator
This is the implementation of the preprocessing and modeling of [Individual tree species prediction using airborne laser scanning data and derived point-cloud metrics within a dual-stream deep learning approach](https://doi.org/10.1016/j.jag.2025.104877).

This paper can be cited as:

```text
@article{Murray2025IndividualTS,
  title={Individual tree species prediction using airborne laser scanning data and derived point-cloud metrics within a dual-stream deep learning approach},
  author={Brent Murray and Nicholas C. Coops and Joanne C. White and Adam Dick and Ignacio Barbeito and Ahmed Ragab},
  journal={Int. J. Appl. Earth Obs. Geoinformation},
  year={2025},
  volume={144},
  pages={104877},
  url={https://api.semanticscholar.org/CorpusID:281619636},
  doi={https://doi.org/10.1016/j.jag.2025.104877}
}
```

Contents
----
```text
├── TreeEstimator/
│   ├── models/
│   │   ├── DGCNN.py
│   │   ├── KPConv.py
│   │   ├── PointExtractor.py
│   │   ├── PointTransformer.py
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
