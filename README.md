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
 - [Base experiment](https://wandb.ai/xevolesi/SqueezeNet/runs/4ynby67q/overview?workspace=user-xevolesi). I was able to get `56.4` top-1 validation accuracy and `79.2` top-5 validation accuracy instead of `57.5` and `80.3` as authors reported. This experiment was interrupted and the plots were captured only for the first 55 epochs instead of 67, иut the weight file you can download is provided after the full training;
 - [Experiment with simple bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/q3wzwk8o/overview?workspace=user-xevolesi). Same hyperparameters; As yoy can see this approach coudn't learn anything;
 - [Experiment with complex bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/5owzmth4/overview?workspace=user-xevolesi). It failed with `CUDA OOM` error with batch size 512. So i performed training with batch size 256. As you can see it came with `nan` for training and validation losses;
 - [Experiment with complex bypass connections but without ReLU in skip-connections](https://wandb.ai/xevolesi/SqueezeNet/runs/5owzmth4/overview?workspace=user-xevolesi).

## SqueezeNet 1.1
 - [Base experiment](https://wandb.ai/xevolesi/SqueezeNet/runs/i9a39wo6/overview?workspace=user-xevolesi). I was able to get `54.6` top-1 validation accuracy and `78.0` top-5 validation accuracy without any changes in hyperparameters. It's really much more efficient without any accuracy drop;
 - [Experiment with simple bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/shqjq8ut/overview?workspace=user-xevolesi). Same hyperparameters. The model achived `53.7` top-1 validation accuracy and `77.3` top-5 validation accuracy;
 - [Experiment with complex bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/aie7p5wn/overview?workspace=user-xevolesi). Failed with `CUDA OOM` error.

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