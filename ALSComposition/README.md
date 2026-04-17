# ALSComposition
This is the implementation of the preprocessing and modeling of [Estimating tree species composition from airborne laser scanning data using point-based deep learning models](https://www.sciencedirect.com/science/article/pii/S0924271623003453?via%3Dihub).

This paper can be cited as:

Murray, B. A., Coops, N. C., Winiwarter, L., White, J. C., Dick, A., Barbeito, I., & Ragab, A. (2024). Estimating tree species composition from airborne laser scanning data using point-based deep learning models. ISPRS Journal of Photogrammetry and Remote Sensing, 207, 282–297. https://doi.org/10.1016/j.isprsjprs.2023.12.008


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
