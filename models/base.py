from typing import Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import BinaryConfusionMatrix

from .feature import ALL_FEATURES
from .data_utils.transforms import VGGish

IGNORE_INDEX = -1 # label for ground truth obstructed by view, ignored in gradient computation
WEIGHTED_LOSS = True  # hard-coding for quick test

class AspedModel(pl.LightningModule):
    # Constructor
    def __init__(
        self,
        exp: Literal["seq2seq", "seq2one"], # Defines whether the model performs seq2seq or seq2one prediction.
        feature: Dict,
        backbone: Dict,
        optim: Dict
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.exp = exp
        self.feature_cfg = feature
        self.backbone_cfg = backbone
        self.optim = optim
        
        # Instantiates the melspectrogram extractor from the ALL_FEATURES dictionary.
        self.mel_spec = ALL_FEATURES[self.feature_cfg["name"]](**self.feature_cfg["args"])

        self.backbone = VGGish(pproc=False, trainable=False, pre_trained=True)
        
        # Initializes confusion matrices for train, validation, and test phases.
        self.init_confusion_matrix()
        # Added for Adam optimizer (Initialize weights with custom stddev)
        self.init_weights(stddev=0.01)
 
        # Define model architecture
        self.segment_length = 10 # Length of input sequence
        self.h_dim = 128 # Hidden dimension of encoder
        self.token_dim = 128 # Transformer input dim
        self.n_classes = 2 # Binary Classification
        self.nEncoders = 1 # Number of transformer blocks

        # Positional Encoding for Trnasformer
        self.pe = torch.nn.Embedding(num_embeddings=self.segment_length, embedding_dim=self.token_dim)
        self.pe_input = torch.Tensor(range(self.segment_length)).long()
        
        transformer_layers = torch.nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=4, dim_feedforward=self.h_dim, dropout=0.2, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(transformer_layers, self.nEncoders)

        self._proj = torch.nn.Sequential(torch.nn.Linear(self.h_dim, self.h_dim), torch.nn.ReLU(), torch.nn.Linear(self.h_dim, self.n_classes))

        self.lr = 0.0005
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index = IGNORE_INDEX)
        
    def weighted_loss(self, X, y):
        '''
        Binary class-weighted loss term. Refer to ICASSP 2024 paper for details
        '''
        idx_pos = torch.argwhere(y > 0)[:, 0]
        idx_neg = torch.argwhere(y == 0)[:, 0]

        try:
            pos_w = min(1, (1/(idx_pos.flatten().shape[0])) / ((1/(idx_pos.flatten().shape[0]) + 1/(idx_neg.flatten().shape[0])) + 0))
        except ZeroDivisionError:
            pos_w = 0
        
        pos_loss = pos_w * self.loss_fn(X[idx_pos].to('cpu'), y[idx_pos].to('cpu'))
        pos_loss = torch.Tensor([0]) if torch.isnan(pos_loss) else pos_loss
        neg_loss = (1 - pos_w) * self.loss_fn(X[idx_neg].to('cpu'), y[idx_neg].to('cpu'))
        neg_loss = torch.Tensor([0]) if torch.isnan(neg_loss) else neg_loss

        return  pos_loss + neg_loss
    
    def loss(self, X, y): # Used in common_step
        y = y.long()
        return self.weighted_loss(X, y)
   
    def forward(self, x):
        """
        :param x: [batch_size x segment_length x 1 x 96 x 64] (encoder type: vggish, conv, vggish-finetune, conv-lite)
                  [batch_size x segment_length x 100 x 128] (encoder type: ast)

        :return output: [batch_size, segment_length, n_classes]
        """    
        # x: torch.Size([256, 10, 1, 96, 64])
        output = self.backbone(x) # MelSpec +  # VGGish  
        output = self.pe(self.pe_input.to(self.device)) + output
        output = self.transformer(output) # output: torch.Size([256, 10, 128])
        output = self._proj(output)
        return output     

    
    def common_step(self, batch):
        X, y = batch
        pred = self.forward(X).permute(0, 2, 1)

        loss_dict = dict()
        loss = self.loss(pred, y)

        loss_dict["loss/pred"] = loss
        loss_dict["loss/total"] = loss

        return loss_dict, pred 
    
    def training_step(self, batch, batch_idx):
        loss_dict, output = self.common_step(batch)
        self.log_dict_prefix(loss_dict, prefix="train", prog_bar=True)
        softmax_output = torch.softmax(output, dim=1)
        class_1_probs = softmax_output[:, 1, :]
        self.train_confusion_matrix.update(class_1_probs, batch[1])
        return loss_dict["loss/total"]

    def validation_step(self, batch, batch_idx):
        loss_dict, output = self.common_step(batch)
        self.log_dict_prefix(loss_dict, prefix="val", prog_bar=True)

        softmax_output = torch.softmax(output, dim=1) 
        class_1_probs = softmax_output[:, 1, :]
        
        self.val_confusion_matrix.update(class_1_probs, batch[1])
        return loss_dict["loss/total"]

    def test_step(self, batch, batch_idx):
        loss_dict, output = self.common_step(batch)
        self.log_dict_prefix(loss_dict, prefix="test")
        softmax_output = torch.softmax(output, dim=1)
        class_1_probs = softmax_output[:, 1, :]
        self.test_confusion_matrix.update(class_1_probs, batch[1])
        return loss_dict["loss/total"]
    
    def on_train_epoch_end(self) -> None:
        confusion_matrix = self.train_confusion_matrix.compute()
        metrics_dict = self.confusion_matrix_to_metrics(confusion_matrix)

        self.log_dict_prefix(metrics_dict, "train")
        self.train_confusion_matrix.reset()

    def on_validation_epoch_end(self) -> None:
        confusion_matrix = self.val_confusion_matrix.compute()
        metrics_dict = self.confusion_matrix_to_metrics(confusion_matrix)

        self.log_dict_prefix(metrics_dict, "val", prog_bar=True)
        self.val_confusion_matrix.reset()

    def on_test_epoch_end(self) -> None:
        # Compute confusion matrix
        confusion_matrix = self.test_confusion_matrix.compute()

        # Unpack confusion matrix values
        ((tn, fp), (fn, tp)) = confusion_matrix
        tn, fp, fn, tp = map(float, (tn, fp, fn, tp))

        # Compute Macro Average Accuracy
        class_0_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0  # Accuracy for class 0
        class_1_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0  # Accuracy for class 1
        macro_accuracy = (class_0_accuracy + class_1_accuracy) / 2

        # Log Macro Average Accuracy
        self.log("test/MacroAccuracy", macro_accuracy, prog_bar=True)

        # Log confusion matrix values (useful for debugging)
        self.log("test/TrueNegative", tn)
        self.log("test/FalsePositive", fp)
        self.log("test/FalseNegative", fn)
        self.log("test/TruePositive", tp)

        # Print confusion matrix for clarity
        print(f"Confusion Matrix:\n"
            f"True Negative (TN): {tn}\n"
            f"False Positive (FP): {fp}\n"
            f"False Negative (FN): {fn}\n"
            f"True Positive (TP): {tp}\n")
        print(f"Macro Accuracy: {macro_accuracy:.4f}")

        # Convert confusion matrix to metrics and log them
        metrics_dict = self.confusion_matrix_to_metrics(confusion_matrix)
        self.log_dict_prefix(metrics_dict, "test")

        # Reset metrics
        self.test_confusion_matrix.reset()

    def log_dict_prefix(self, d, prefix, prog_bar=False):
        for k, v in d.items():
            self.log("{}/{}".format(prefix, k), v, prog_bar=prog_bar)

    # Confusion Matrix Handling
    ## Intializes binary confusion matrices for tracking performance.
    def init_confusion_matrix(self):
        self.train_confusion_matrix = BinaryConfusionMatrix()
        self.val_confusion_matrix = BinaryConfusionMatrix()
        self.test_confusion_matrix = BinaryConfusionMatrix()
        
    ## Converts the confusion matrix into precision, recall, and F1 metrics for both positive and negative classes.
    def confusion_matrix_to_metrics(self, confusion_matrix):
        ((tn, fp), (fn, tp)) = confusion_matrix
        print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
        tn, fp, fn, tp = map(float, (tn, fp, fn, tp))
        # p = precision, r = recall, f = F1 score
        p_n = tn / (tn + fn) if (tn + fn) > 0 else 0
        r_n = tn / (tn + fp) if (tn + fp) > 0 else 100
        f_n = 2 * p_n * r_n / (p_n + r_n) if (p_n + r_n) > 0 else 0

        p_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_p = tp / (tp + fn)
        f_p = 2 * p_p * r_p / (p_p + r_p) if (p_p + r_p) > 0 else 0

        return dict(
            PositivePrecision=p_p, PositiveRecall=r_p, PositiveF1=f_p,
            NegativePrecision=p_n, NegativeRecall=r_n, NegativeF1=f_n,
            MacroPrecision=(p_p+p_n) / 2, MacroRecall=(r_p+r_n) / 2, MacroF1=(f_p+f_n) / 2
        )

    # Optimizer Configuration
    #   - Dynamically loads the optimizer type from torch.optim
    #   - Configures the optimizer with parameters from the YAML config.
    def configure_optimizers(self):
        optim_args = self.optim["args"]
        Optimizer = torch.optim.__dict__[self.optim["type"]]  # Load the appropriate optimizer
        
        # Create the optimizer with the specified arguments
        optimizer = Optimizer(self.parameters(), **optim_args)

        # Handle TensorBoard or other loggers
        if self.logger:
            if isinstance(self.logger.experiment, torch.utils.tensorboard.writer.SummaryWriter):
                # TensorBoard logging
                self.logger.experiment.add_scalar("optimizer/lr", optim_args["lr"], 0)
                if self.optim["type"] == "Adam":
                    self.logger.experiment.add_scalar("optimizer/eps", optim_args.get("eps", 1e-8), 0)
            else:
                # For other loggers (e.g., W&B)
                self.logger.experiment.log({
                    "optimizer/type": self.optim["type"],
                    "optimizer/lr": optim_args["lr"],
                    "optimizer/eps": optim_args.get("eps", 1e-8) if self.optim["type"] == "Adam" else None,
                })

        return optimizer


    # Added for Adam optimizer
    def init_weights(self, stddev=0.01):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=stddev)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
