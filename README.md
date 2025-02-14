# PedestrianDetection
Urban Audio Sensing Project @ Georgia Tech (Fall2024-Spring2025)
[**[Paper]**](https://arxiv.org/abs/2309.06531)
[**[Website]**](https://urbanaudiosensing.github.io/)

## Environment Setup
After cloning the repository, change the working directory to this repository by using the `cd` command and then enter the below Linux command on your terminal.
```bash
conda create -y -n my_env_name python=3.9.18
conda activate my_env_name
sh download_dependencies.sh
```
## Download Dataset

## How to train
Ensure the dataset paths in the YAML files are well-mapped!
### Train w/ ASPED v.a
```bash
python main.py train --config="configs/vggish_finetune_ASPED-a.yaml"
```
### Train w/ ASPED v.b
```bash
python main.py train --config="configs/vggish_finetune_ASPED-b.yaml"
```

## How to test
```bash
python main.py test  --config="/path/to/your/config.yaml" --checkpoint_path="/path/to/your/checkpoint.ckpt"
```
