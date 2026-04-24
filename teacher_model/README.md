## Downloading Teacher Models

In our experiments, we use two pretrained robust teacher models from the RobustBench model zoo. 
Both teacher models are based on the WideResNet-34-10 architecture and are pretrained on CIFAR-10.

The two teacher models that need to be downloaded are:

| Teacher model | Architecture | Google Drive ID |
|---|---|---|
| Chen2021LTD_WRN34_10 | WideResNet-34-10 | `1-0RoQKYvHLNh7hZ71wJjSit1XtrJQo9D` |
| Cui2023Decoupled_WRN-34-10 | WideResNet-34-10 | `1-ArD-TugRXUbH3VtM9qnzvby6NvdXNUN` |

The checkpoints can be downloaded manually using `gdown`.

```bash
pip install gdown

mkdir -p teacher_model/cifar10

gdown "https://drive.google.com/uc?id=1-0RoQKYvHLNh7hZ71wJjSit1XtrJQo9D" \
    -O teacher_model/cifar10/Chen2021LTD_WRN34_10.pt

gdown "https://drive.google.com/uc?id=1-ArD-TugRXUbH3VtM9qnzvby6NvdXNUN" \
    -O teacher_model/cifar10/Cui2023Decoupled_WRN-34-10.pt