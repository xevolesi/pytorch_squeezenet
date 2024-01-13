# pytorch_squeezenet
Implementation of [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4) by `Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer`.

# Results

## Paper version
To reproduce the article I mainly focused on the hyperparameters in [this](https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/solver.prototxt) `.prototxt` file from paper's authors. In addition i used:

1. Weights initialization described in paper;
2. There is not much information about image preprocessing in paper, but there are some issues with author's answers about image preprocessing and augmentation ([link 1](https://github.com/forresti/SqueezeNet/issues/8), [link2](https://github.com/forresti/SqueezeNet/issues/62)). There is also `.prototxt` file with only mean substraction, even without division by 255. So, i used the author's answers to build augmentation and preprocessing pipeline with [albumentation](https://github.com/albumentations-team/albumentations);
3. I used `torch.optim.lr_scheduler.PolynomialLR` and do it's step __after each batch__. As far as i understand in `Caffe` it works like this;
4. I didn't use any gradient accumulation techniques as described in `.prototxt` and directly trained using `batch_size=512`;
5. I also tried to calculate how many epochs i need to train to match `170_000` batches as described in `.prototxt` file and got `~67` epochs to train.

| Architecture  | Acc@1 | Acc@5 | Skip connections type | W&B run |			PyTorch checkpoints			        |
|---------------|------ |-------|-----------------------|---------|--------------------------------------|
| SqueezeNet 1.0| 56.0  |79.0   |None                   |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/89e3ebaw/overview?workspace=user-xevolesi)|[link](https://drive.google.com/file/d/1LHo8aNxk0KYZj_EDqu_2sptr1-WgSiIg/view?usp=sharing)|
| SqueezeNet 1.0| 51.3  |74.8   |simple                 |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/4b8c64rt/overview?workspace=user-xevolesi)|[link](https://drive.google.com/file/d/1sLqRwhEmgZScXQKXk3a8Y_8aMg3qR7qZ/view?usp=sharing)|
| SqueezeNet 1.0| NaN   |NaN    |complex                |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/iikz83kq/overview?workspace=user-xevolesi)|-|
| SqueezeNet 1.1| 54.6  |78.0   |None                   |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/i9a39wo6/overview?workspace=user-xevolesi)|[link](https://drive.google.com/file/d/1Or3PmI_BX0yFz0qEHbXhKH8r0ht33Q_d/view?usp=sharing)|
| SqueezeNet 1.1| 53.7  |77.3   |simple                 |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/shqjq8ut/overview?workspace=user-xevolesi)|[link](https://drive.google.com/file/d/1Ra7lMbXRr5RKL8rx8kW4dtt3S_fdLZjV/view?usp=sharing)|
| SqueezeNet 1.1| NaN   |NaN    |complex                |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/iikz83kq/overview?workspace=user-xevolesi)|-|

It was also interesting for me to play a bit with `SqueezeNets`. So i decide to add the following things:
- In `ExpandLayer` use only one `torch.nn.ReLU` instead of two. Obviously, it's faster;
- Swap first `torch.nn.MaxPoll2d` with first `torch.nn.ReLU`. This will lead to applying ReLU to the tensor with much less data which is much faster;
- Substitute `final_conv` with single `torch.nn.Linear` layer, so we don't have convolution + ReLU for the final classification layer wich was quite strange decision since ReLU killed all logits that are below zero;
- Standard PyTorch initialization for layers;
- Add `BatchNorm2d` to the model;
- Use standard modern Z-normalization for images (division by 255 and Z-standardization with ImageNet means and stds);
- Use `torch.optim.lr_scheduler.CosineAnnealingLR` instead of `torch.optim.lr_scheduler.PolynomialLR`;
- `RandomResizedCrop(227, 227)` instead of `Compose(Resized(256, 256), RandomResizedCrop(227, 227))`;
- Add `SE-module`.

| Architecture  | Acc@1 | Acc@5 | Skip connections type | W&B run |			PyTorch checkpoints			        |
|---------------|------ |-------|-----------------------|---------|--------------------------------------|
| SqueezeNet 1.2| 61.8  |84.6   |None                   |[link](https://wandb.ai/xevolesi/SqueezeNet/runs/tg2yt980/overview?workspace=user-xevolesi)|[link](https://drive.google.com/file/d/1nt185v8q78RB6xyrmT8aosb30hprI9xo/view?usp=sharing)|


# Dataset
I used ImageNet from `Kaggle`. You can find it [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview).

# Installation
## For training
I used the following environment configuration:
- Ubuntu 22.04;
- Python 3.11;
- Python 3.11-venv;
- NVidia Driver 530.41.03;
- NVidia CUDA 12.2

Core requirements are listed in `requirements.txt` file. Development requirements are listed in `requirements.dev.txt`. I also provide to you optional requirements in `requirements.optional.txt`. So, to be able to use this repo you need to do the following things:
1. Install `python 3.11`, `python3.11-dev` and `python3.11-venv`:
	```
	sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
	```
2. Install `libturbojpeg`. You need it to use `jpeg4py` package which greatly reduce the time of reading `.jpeg\.jpg` images from disk:
	```
	sudo apt-get install libturbojpeg
	```
3. Create `venv`, activate it and update `pip`:
	```
	python3.11 -m venv venv
	source venv/bin/activate
	pip install --upgrade pip
	```
4. Install core dependencies:
	```
	pip install -r requirements.txt
	```
5. If you want to use `W&B` logger then install `requirements.optional.txt` by `pip install -r requirements.optional.txt`. And create `.env` file and put `WANDB_API_KEY` inside `.env`. You can find example in `.env-example`.

## For development
If you want to further develop this repo you need to install development requirements. Development requirements contain `ruff` package for linting\formatting and `pytest` packages for tests.

## Makefile

I also provide to you some recipies in `Makefile` to make your life easier:
- `make lint` - runs `ruff` linter;
- `make format` - runs `ruff` formatter;
- `make run_tests` - runs tests with `pytest`;
- `make pre_push_test` - runs linter and tests.

These recipies are mostly for development purposes.

# How to train
It's quite simple:
- Modify `config.yml` file according to your desires;
- Run `python train.py`.

It shoud works okay. I also provide to you some predefined configuration files placed in `predefined_configs`. These are configuration files that i used to obtain models listed in tables above.

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