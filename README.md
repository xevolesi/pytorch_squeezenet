# pytorch_squeezenet
Implementation of [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4) by `Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer`.

# Reproduction
To reproduce the article I mainly focused on the hyperparameters in [this](https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/solver.prototxt) `.prototxt` file from paper's authors. In addition i used:

1. Weights initialization described in paper;
2. Image normalization from `.prototxt` file (only mean substraction, even without division by 255);
3. Almost no augmentations, except `RandomResizedCrop`. I didn't find any mention of any augmentations in paper or in `.prototxt` file;
4. I didn't use any gradient accumulation techniques as described in `.prototxt` and directly trained using `batch_size=512`.`
5. I also tried to calculate how many epochs i need to train to match `170_000` batches as described in `.prototxt` file and got `~67` epochs to train.

## SqueezeNet 1.0
 - [Base experiment](https://wandb.ai/xevolesi/SqueezeNet/runs/4ynby67q/overview?workspace=user-xevolesi). I was able to get `56.4` top-1 validation accuracy and `79.2` top-5 validation accuracy instead of `57.5` and `80.3` as authors reported. This experiment was interrupted and the plots were captured only for the first 55 epochs instead of 67, иut the weight file you can download is provided after the full training;
 - [Experiment with simple bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/q3wzwk8o/overview?workspace=user-xevolesi). Same hyperparameters; As yoy can see this approach coudn't learn anything;
 - [Experiment with complex bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/5owzmth4/overview?workspace=user-xevolesi). It failed with `CUDA OOM` error with batch size 512. So i performed training with batch size 256. As you can see it came with `nan` for training and validation losses;
 - [Experiment with complex bypass connections but without ReLU in skip-connections](https://wandb.ai/xevolesi/SqueezeNet/runs/5owzmth4/overview?workspace=user-xevolesi).

## SqueezeNet 1.1
 - [Base experiment](https://wandb.ai/xevolesi/SqueezeNet/runs/ebieb0gy/overview?workspace=user-xevolesi). I was able to get `56.1` top-1 validation accuracy and `79.0` top-5 validation accuracy without any changes in hyperparameters. It's really much more efficient without any accuracy drop;
 - [Experiment with simple bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/emyg6tuz/overview). Same hyperparameters. The model achived almost the same top-1 and top-5 accuracies as the base model;
 - [Experiment with complex bypass connections](https://wandb.ai/xevolesi/SqueezeNet/runs/nhnaw96z/overview?workspace=user-xevolesi). Same hyperparameters; As yoy can see this approach coudn't learn anything;

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