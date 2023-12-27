# pytorch_squeezenet
Implementation of [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4) by `Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer`.

# Reproduction
To reproduce the article I mainly focused on the hyperparameters in [this](https://github.com/forresti/SqueezeNet/blob/master/SqueezeNet_v1.0/solver.prototxt) `.prototxt` file from paper's authors. In addition i used:

1. Weights initialization described in paper;
2. Image normalization from `.prototxt` file (only mean substraction, even without division by 255);
3. Almost no augmentations, except `RandomResizedCrop`. I didn't find any mention of any augmentations in paper or in `.prototxt` file;
4. I didn't use any gradient accumulation techniques as described in `.prototxt`.
5. I also tried to calculate how many epochs i need to train to match `170_000` batches as described in `.prototxt` file and got `~67` epochs to train.

Here is the results of reproduction:
1. Simple SqueezeNet (without skip-connections): https://wandb.ai/xevolesi/SqueezeNet/runs/xk9u0653/overview?workspace=user-xevolesi. I was able to get `56.6` top-1 validation accuracy instead of `57.5` as authors reported. But as you can see there is quite obvious increasing trend for top-1 and top-5 validation accuracies so i think that if i train longer i can easilly achieve authors results or even more;
2. SqueezeNet with simple bypass connections:

    2.1 I was not able to reproduce the results with bypass connections. Here is the [experiment with the same hyperparameters](https://wandb.ai/xevolesi/SqueezeNet/runs/vfi0r453/overview?workspace=user-xevolesi);

    2.2 I tried different learning rates: [LR=0.03](https://wandb.ai/xevolesi/SqueezeNet/runs/42usodxu/overview?workspace=user-xevolesi), [LR=0.02](https://wandb.ai/xevolesi/SqueezeNet/runs/jky2j69o/overview?workspace=user-xevolesi), [LR=0.01](https://wandb.ai/xevolesi/SqueezeNet/runs/q42atfuk/overview?workspace=user-xevolesi), but the result was sligtly worse than without simple bypass connections.


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