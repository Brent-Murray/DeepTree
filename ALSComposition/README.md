# ALSComposition
This is the implementation of the preprocessing and modeling of [Estimating tree species composition from airborne laser scanning data using point-based deep learning models](https://www.sciencedirect.com/science/article/pii/S0924271623003453?via%3Dihub).

This paper can be cited as:

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

Contents
----
```text
├── ALSComposition/
│   ├── augment/
│   │   ├── __init__.py
│   │   └── augmentor.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── loss_utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dgcnn.py
│   │   └── pointnet2.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── augmentation.py
│   │   ├── send_telegram.py
│   │   ├── tools.py
│   │   └── train.py
│   ├── main.py
│   └── README.md
```
