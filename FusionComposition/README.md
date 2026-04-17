# FusionComposition 
This is the implementation of the preprocessing and modeling of [Tree speices proportion prediction using airborne laser scanning and Sentinel-2 data within a deep learning based dual-stream data fusion approach](https://doi.org/10.1080/01431161.2025.2521072).

This paper can be cited as:

Murray, B. A., Coops, N. C., White, J. C., Dick, A., & Ragab, A. (2025). Tree species proportion prediction using airborne laser scanning and Sentinel-2 data within a deep learning based dual-stream data fusion approach. International Journal of Remote Sensing, 46(14), 5436–5464. https://doi.org/10.1080/01431161.2025.2521072



Contents
----
```text
├── FusionComposition/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dgcnn.py
│   │   ├── ensamble_unet.py
│   │   ├── fusion_unet.py
│   │   ├── NEWretain_unet.py
│   │   ├── retain_unet.py
│   │   └── unet.py
│   ├── utils/
│   │   ├── install_packages/
│   │   │   ├── install_packages.sh
│   │   │   ├── install_packages2.sh
│   │   │   └── README.txt
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── lidRProcessing.R
│   │   ├── loss_utils.py
│   │   ├── parallel_fps.py
│   │   ├── send_telegram.py
│   │   ├── tools.py
│   │   └── train.py
│   ├── main.py
│   └── README.md
```
