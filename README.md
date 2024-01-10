# pytorch_squeezenet
Implementation of [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4) by `Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer`.

# Reproduction
To reproduce the article I mainly focused on the hyperparameters in [this](https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/solver.prototxt) `.prototxt` file from paper's authors. In addition i used:

1. Weights initialization described in paper;
2. There is not much information about image preprocessing in paper, but there are some issues with author's answers about image preprocessing and augmentation ([link 1](https://github.com/forresti/SqueezeNet/issues/8), [link2](https://github.com/forresti/SqueezeNet/issues/62)). There is also `.prototxt` file with only mean substraction, even without division by 255. So, i used the author's answers to build augmentation and preprocessing pipeline with [albumentation](https://github.com/albumentations-team/albumentations);
3. I used `torch.optim.lr_scheduler.PolynomialLR` and do it's step __after each batch__. As far as i understand in `Caffe` it works like this;
4. I didn't use any gradient accumulation techniques as described in `.prototxt` and directly trained using `batch_size=512`;
5. I also tried to calculate how many epochs i need to train to match `170_000` batches as described in `.prototxt` file and got `~67` epochs to train.

## SqueezeNet 1.0
 - [Base experiment](https://wandb.ai/xevolesi/SqueezeNet/runs/89e3ebaw/overview?workspace=user-xevolesi). I was able to get `56.0` top-1 validation accuracy and `79.0` top-5 validation accuracy instead of `57.5` and `80.3` as authors reported;
 - [Experiment with simple bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/4b8c64rt/overview?workspace=user-xevolesi). Same hyperparameters. The model achived `51.3` top-1 validation accuracy and `74.8` top-5 validation accuracy;
 - [Experiment with complex bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/iikz83kq/overview?workspace=user-xevolesi). Filed with `CUDA OOM` error. I tired to train with batch size = 384 ([link](https://wandb.ai/xevolesi/SqueezeNet/runs/zv7ap2ar)) but the loss was `nan`.

## SqueezeNet 1.1
 - [Base experiment](https://wandb.ai/xevolesi/SqueezeNet/runs/i9a39wo6/overview?workspace=user-xevolesi). I was able to get `54.6` top-1 validation accuracy and `78.0` top-5 validation accuracy without any changes in hyperparameters. It's really much more efficient without any accuracy drop;
 - [Experiment with simple bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/shqjq8ut/overview?workspace=user-xevolesi). Same hyperparameters. The model achived `53.7` top-1 validation accuracy and `77.3` top-5 validation accuracy;
 - [Experiment with complex bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/iikz83kq/overview?workspace=user-xevolesi). Doesn't learn anything.

# SqueezeNet 1.2
This is the model with some bag of freebies.
- In `ExpandLayer` use only one `torch.nn.ReLU` instead of two. Obviously, it's faster;
- Swap first `torch.nn.MaxPoll2d` with first `torch.nn.ReLU`. This will lead to applying ReLU to the tensor with much less data which is much faster;
- Substitute `final_conv` with single `torch.nn.Linear` layer, so we don't have convolution + ReLU for the final classification layer wich was quite strange decision since ReLU killed all logits that are below zero;
- Standard PyTorch initialization for layers;
- Add `BatchNorm2d` to the model;
- Use standard modern Z-normalization for images (division by 255 and Z-standardization with ImageNet means and stds);
- Use `torch.optim.lr_scheduler.CosineAnnealingLR` instead of `torch.optim.lr_scheduler.PolynomialLR`;
- `RandomResizedCrop(227, 227)` instead of `Compose(Resized(256, 256), RandomResizedCrop(227, 227))`;
- Add `SE-module`.

# How to use
1. Install `python 3.11`, `python3.11-dev` and `python3.11-venv`:
```
sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
```
2. Create `venv`:
```
python3.11 -m venv venv
```
3. Activate `venv`:
```
source venv/bin/activate
```
4. Update `pip`:
```
pip install --upgrade pip
```
5. Install dependencies:
```
pip install -r requirements.txt
```
6. If you want to use `W&B` logger then install `requirements.optional.txt` by `pip install -r requirements.optional.txt`. And create `.env` file and put `WANDB_API_KEY` inside `.env`. You can find example in `.env-example`.
7. Fill in `config.yml` file according to your settings.
8. Run `python train.py`

# Export to ONNX
To export the model just do:

```
python export_to_onnx.py \
		--config %path to config.yml% \
		--torch_weights %path to PyTorch weights% \
		--onnx_path %path tot ONNX file% \
		--image_size %Image height and image width separated by single comma%
```

Example of command to export `squeezenet-v1.1`:

```
python export_to_onnx.py \
		--config predefined_configs/squeezenetv1.1.yml \
		--torch_weights weights/i9a39wo6/state_dict.pth \
		--onnx_path ./squeezenetv11.onnx \
		--image_size 224,224 \
```