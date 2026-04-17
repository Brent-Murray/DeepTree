# DeepTree
Deep learning approaches for tree species estimation from remote sensing data.

This repository contains code, and workflows for using airborne and satellite remote sensing data.

# Repository Structure
The repository is organized into four main sections, each corresponding to a specific study:

```text
├── ALSComposition/
├── FusionComposition/
├── LLPEstimator/
├── TreeEstimator/
```

Each folder includes code, data processing pipelines, and model implementations used in the associated work.

## Studies
- **ALSComposition** - *Estimating tree species composition from airborne laser scanning data using point-based deep learning models*
- **FusionComposition** - *Tree species proportion prediction using airborne laser scanning and Sentinel-2 data within a deep learning based dual-stream data fusion approach*
- **LLPEstimator** - *Using weakly supervised deep learning to derive individual tree species and plot-level species composition from airborne laser scanning data*
- **TreeEstimator** - *Individual tree species prediction using airborne laser scanning data and derived point-cloud metrics within a dual-stream deep learning approach*

## Data
Datasets are not included due to size and/or licensing constraints.

## Citation
If you use this repository, please cite the corresponding publications:

**ALSComposition**
```text
@article{Murray2024EstimatingTS,
  title={Estimating tree species composition from airborne laser scanning data using point-based deep learning models},
  author={Brent Murray and Nicholas C. Coops and Lukas Winiwarter and Joanne C. White and Adam Dick and Ignacio Barbeito and Ahmed Ragab},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:266711000},
  doi={https://doi.org/10.1016/j.isprsjprs.2023.12.008}
}
```

**FusionComposition**
```text
@article{Murray2025TreeSP,
  title={Tree species proportion prediction using airborne laser scanning and Sentinel-2 data within a deep learning based dual-stream data fusion approach},
  author={Brent Murray and Nicholas C. Coops and Joanne C. White and Adam Dick and Ahmed Ragab},
  journal={International Journal of Remote Sensing},
  year={2025},
  volume={46},
  pages={5436 - 5464},
  url={https://api.semanticscholar.org/CorpusID:279559010},
  doi={https://doi.org/10.1080/01431161.2025.2521072}
}
```

**LLPEstimator**
```text
Will update upon acceptance of publication
```

**TreeEstimator**
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

## Notes
This repository is actively maintained and may evolve as additional experiments and studies are completed
