# Computer Vision Portfolio

A portfolio of computer vision projects covering image classification, object detection, image processing, and generative AI. Projects are developed independently and documented in their own folders.

## Featured project

| Project | Overview |
| --- | --- |
| [MNIST Digit Classification](./projects/mnist-pytorch/README.md) | PyTorch CNN project with reproducible training, held-out evaluation, single-image prediction, visualized results, and automated tests. |

The `projects/` folder is for current portfolio highlights. As work progresses, will move other projects there and for now less-active works are in `archive/`folder.

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
│   └── workflows/   # MNIST CI workflow
├── projects/
│   └── mnist-pytorch/                    # Featured digit-classification project                
├── archive/
│   ├── README.md
│   ├── distracted-driver-detection/
│   ├── image-processing-object-distance/
│   └── stable-diffusion/
└── README.md
```

Each project has its own dependencies and setup instructions. See its README before running its code.
