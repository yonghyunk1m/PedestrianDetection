# PedestrianDetection
Urban Audio Sensing Project @ Georgia Tech (Fall2024-Spring2025)
[**[Paper]**](https://arxiv.org/abs/2309.06531)
[**[Website]**](https://urbanaudiosensing.github.io/)

## Environment Setup
After cloning the repository, change the working directory to this repository by using the `cd` command and then enter the below Linux command on your terminal.
```bash
sh prerequisites.sh
```
## How to train
```bash
python main.py train --config="/path/to/your/config.yaml"
```

## How to test
```bash
python main.py test  --config="/path/to/your/config.yaml" --checkpoint_path="/path/to/your/checkpoint.ckpt"
```
