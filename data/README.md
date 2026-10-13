# 📊 Dataset Documentation & Guidelines

## Required Dataset Layout

The dataset used for training and evaluating the Deepfake Image Detection models should be structured in the following directory hierarchy:

```text
dataset/
├── train/
│   ├── real/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── fake/
│       ├── image_001.jpg
│       └── ...
├── validation/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Dataset Configuration

Do not place large datasets inside Git repository. Configure dataset paths using `ml/configs/default.yaml` or environment variables:

```yaml
dataset:
  path: "/path/to/your/dataset"
  image_size: 224
  batch_size: 32
  num_workers: 4
```

## Supported Datasets
- **FaceForensics++**
- **DeepFake Detection Challenge (DFDC)**
- **Celeb-DF**
- **Custom Synthetic (Midjourney / Stable Diffusion / StyleGAN) vs Authentic Datasets**
