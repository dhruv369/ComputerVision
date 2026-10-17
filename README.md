# Computer Vision Portfolio

A portfolio of computer vision projects covering image classification, object detection, image processing, and generative AI. Projects are developed independently and documented in their own folders.

## Featured project

| Project | Overview |
| --- | --- |
| [MNIST Classification](./projects/mnist-keras/mnist_readme.md) | PyTorch CNN experiments with MNIST and Fashion-MNIST, training metrics, a comparison dashboard, and CI testing. |

The `projects/` folder is for current portfolio highlights. As work progresses, feature one or two projects there and move completed or less-active work to `archive/`.

## Archived projects

| Project | Overview |
| --- | --- |
| [Image Processing and Object Distance](./archive/image-processing-object-distance/Source.cpp) | C++/OpenCV image-processing work focused on estimating object distance. |
| [Distracted Driver Detection](./archive/distracted-driver-detection/README.md) | Deep-learning project for classifying distracted-driver behavior. |
| [Stable Diffusion Experiment](./archive/stable-diffusion/stable_diffusion_xl_1_0_inpainting_0_1.ipynb) | Notebook exploring Stable Diffusion XL inpainting. |

## Repository structure

```text
ComputerVision/
├── .gitignore
├── .github/
│   └── workflows/                       # MNIST CI workflow
├── archive/
│   ├── README.md
│   ├── distracted-driver-detection/
│   ├── image-processing-object-distance/
│   └── stable-diffusion/
├── projects/
│   └── mnist-keras/                      # Featured classification project
└── README.md
```

Each project has its own dependencies and setup instructions. See its README before running its code.
