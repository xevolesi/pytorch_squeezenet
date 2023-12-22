# pytorch_squeezenet
Implementation of [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360v4) by `Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer`.

# Results

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