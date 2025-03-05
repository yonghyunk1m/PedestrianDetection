import yaml
import torch
torch.set_float32_matmul_precision('high')
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

'''
DESCRIPTION
- yaml: To load the configuration file (config.yaml), which stores training hyperparameters and settings.
- torch: Core PyTorch library for building models and data pipelines
- torch.set_float32_matmul_precision('high'):  Optimizes matrix multiplication operations for better performance.
- pytorch_lightning: A high-level library that simplifies training workflows in PyTorch by handling boilerplate code.
- TensorBoardLogger: Logs metrics and information to TensorBoard, a visualization tool.
- WandbLogger: Logs experiment details to Weights & Biases (W&B), a popular experiment tracking tool.
- ModelCheckpoint: A callback that saves the model based on specific criteria during training.
'''

from models.data_utils.dataset import ASPEDDataset, SR
from models.data_utils.datamodule import AspedDataModule
from models import ALL_MODELS # A dictionary that holds different model architectures, allowing dynamic model loading.
import wandb

# The train function is the main function where model training occurs.
def train(config="/home/yding/aspad-models/configs/vggish_finetune_yonghyun_FEB12.yaml", checkpoint_path=None):
    
    # OPENS the configuration YAML file and LOADS it into the config dictionary using yaml.safe_load
    # This dictionary contains settings for the data module, model, and trainer.
    with open(config) as f:
        config = yaml.safe_load(f)

    # Sets a global random seed for reproducibility.
    pl.seed_everything(config["seed"], workers=True) # workers=True: Ensures that multi-threaded operations (like data loaders) are also seeded.

    # Extract configurations
    data_cfg = config["data"] # Configuration for the data module.
    model_cfg = config["model"] # Config for the model.
    trainer_cfg = config["trainer"] # Config for the trainer and callbacks.

    TEST_DIR = config['data_params'].pop('data_path') 
    print(f"TEST_DIR: {TEST_DIR}")
    
    # Initialize DataModule and Model
    # PAVAN Load dataset
    X = ASPEDDataset.from_dirs_v1(TEST_DIR, **config['data_params'])
    datamodule = AspedDataModule(X, **config['dataloader_params']) # PyTorch Lightning의 DataModule; train/val/test_dataloader()제대로 설정해야함

    model = ALL_MODELS[model_cfg.pop("type", "base")](exp=config["exp"], **model_cfg) # Dynamically selects a model from the ALL_MODELS dictionary based on the "type" field from model_cfg.
    # base: AspedModel || teacher_student: TeacherStudnetModel || video: VideoDistillationModel. (see models/__init__.py)

    # # Initialize WandbLogger
    wandb_logger = WandbLogger(
        project="urban_yiwei",  # Name of the project in W&B.
        name=f"{config['exp']}_{trainer_cfg['args']['max_epochs']}_epochs(vggish_finetune_yonghyun_FEB14_6m)",  #  A unique experiment name including the experiment ID and number of epochs.
        config=config  # Logs the configuration to W&B for reference.
    )

    # Initialize TensorBoardLogger:Initializes TensorBoard for logging metrics during training.
    tensorboard_logger = TensorBoardLogger(**trainer_cfg["logger"]) # Uses parameters from the logger section of trainer_cfg.

    # Initializes the PyTorch Lightning Trainer, which handles the training loop.
    trainer = pl.Trainer(
        **trainer_cfg["args"],
        logger= [tensorboard_logger, wandb_logger], #[tensorboard_logger],  # Uses both TensorBoard and W&B for logging.
        callbacks=[
            ModelCheckpoint(**trainer_cfg["checkpoint"]),
        ] # Adds a ModelCheckpoint callback to save model checkpoints according to the criteria defined in trainer_cfg["checkpoint"]
    )
    '''
    checkpoint:
        dirpath: *save_dir
        filename: epoch={epoch}-loss={val/loss/total:.3f}
        auto_insert_metric_name: false
        monitor: val/loss/total
        mode: min
        every_n_epochs: 1
    '''
    
    # Train and Test the Model
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path) # Trains the model using the specified data module.
    trainer.test(datamodule=datamodule, ckpt_path="best") # Tests the model using the best checkpoint from training

# /media/backup_SSD/PedestrianDetection/configs/vggish_finetune_ASPED-a.yaml (test w/ v.b)
# /media/backup_SSD/PedestrianDetection/configs/vggish_finetune_ASPED-b.yaml (test w/ v.a)
def test(config="/media/backup_SSD/PedestrianDetection/configs/vggish_finetune_ASPED-b.yaml", checkpoint_path="/home/yding/aspad-models/work_dir_A3/vggish_finetune_yonghyun_DEC11_1/epoch=17-loss=0.576.ckpt"):
    import copy
    
    with open(config) as f:
        config = yaml.safe_load(f)

    pl.seed_everything(config["seed"], workers=True)

    # Extract configurations
    data_cfg = config["data"]
    model_cfg = config["model"]

    TEST_DIR = config['data_params'].pop('data_path')
    print(f"TEST_DIR: {TEST_DIR}")

    # Initialize DataModule
    X = ASPEDDataset.from_dirs_v1(TEST_DIR, **config['data_params'])
    datamodule = AspedDataModule(X, **config['dataloader_params'])

    # Initialize the model
    model = ALL_MODELS[model_cfg.pop("type", "base")](exp=config["exp"], **model_cfg)

    # Initialize Trainer for testing
    trainer = pl.Trainer()

    # Evaluate over different values of min_val
    eval_dict = dict()
    params = copy.deepcopy(config)  # Save the original configuration for reference
    for tt in [1, 2, 3, 4]:
        ASPEDDataset.min_val = tt  # Update min_val for this iteration
        print(f"Testing with ASPEDDataset.min_val = {tt}")

        # YONGHYUN
        # 🚨 테스트 시 bus frame이 포함된 데이터 건너뛰기
        valid_samples = []
        for idx in range(len(X)):
            batch = X[idx]
            if batch is None:
                continue  # bus frame이 포함된 경우 건너뛰기
            valid_samples.append(batch)
            
        # Run the testing phase
        print(f"Run the testing phase")
        datamodule = AspedDataModule(valid_samples, **config['dataloader_params'])
        m = trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
        eval_dict[str(tt)] = m[0]  # Assuming m is a list of metrics dictionaries

    # Print results for debugging
    print("Evaluation Results:", eval_dict)
    print("Configuration Parameters:", params)

    # Return the results
    return {
        "EVAL": eval_dict,
        "CONFIG": params,
        "MODEL_PATH": checkpoint_path,
    }


if __name__ == "__main__":
    # fire: A library that converts Python functions into command-line interfaces (CLIs).
    import fire 
    fire.Fire({"train": train, "test": test})

'''
HOW TO TRAIN / TEST?

[TRAIN]
python main.py train --config="/path/to/your/config.yaml"

[TEST]
python main.py test  --config="/path/to/your/config.yaml" --checkpoint_path="/path/to/your/checkpoint.ckpt"
(Example)
python main.py test --config="/media/backup_SSD/PedestrianDetection/configs/vggish_finetune_ASPED-b.yaml"\
    --checkpoint_path="/media/backup_SSD/PedestrianDetection/work_dir/vggish_finetune_ASPED-b/epoch=13-loss=0.584.ckpt"
'''