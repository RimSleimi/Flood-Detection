import gc
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from monai.networks.nets import DynUNet, AttentionUnet
from loss import *
from metrics import *



class Unet(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.build_model()
        self.loss = LossFlood()
        self.metrics = MetricsFlood(n_class=self.args.out_channels)
        
    
    def training_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        logits = self.model(img)
        loss = self.loss(logits, lbl)
        self.metrics.update(logits, lbl, loss) 

    def predict_step(self, batch, batch_idx):
        img, lbl = batch
        preds = self.model(img)
        preds = (nn.Sigmoid()(preds) > 0.5).int()
        lbl_np = lbl.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        np.save(self.args.save_path + 'predictions.npy', preds_np)
        np.save(self.args.save_path + 'labels.npy', lbl_np)        

    def training_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        gc.collect()
        
    def validation_epoch_end(self, outputs):
        dice, IoU, precision, recall, omission, comission, loss = self.metrics.compute()
        dice_mean = dice.mean().item()
        IoU_mean = IoU.mean().item()
        mean_precision = precision.mean().item()
        mean_recall = recall.mean().item()
        omission_mean = omission.mean().item()
        comission_mean = comission.mean().item()
        self.metrics.reset()
        
        print(f"Val_Performace: dice_mean {dice_mean:.3f}| IoU_mean {IoU_mean:.3f}| omission_mean {omission_mean:.3f}| comission_mean {comission_mean:.3f}| ,Val_Loss {loss.item():.3f}")
        self.log("dice_mean", dice_mean)
        self.log("Val_Loss", loss.item())
        
        torch.cuda.empty_cache()
        gc.collect()        
        
        
    def build_model(self):

        if self.args.model == 'UNet':
            self.model = smp.Unet(
                encoder_name=self.args.encoder,  
                encoder_weights=None, 
                decoder_use_batchnorm=True, 
                decoder_attention_type=self.args.attention,
                in_channels=self.args.in_channels,
                classes=self.args.out_channels)
        
        elif self.args.model == 'UNet++':
            self.model = smp.UnetPlusPlus(
                encoder_name=self.args.encoder, 
                encoder_weights=None,
                decoder_use_batchnorm=True,  
                decoder_attention_type=self.args.attention,
                in_channels=self.args.in_channels,
                classes=self.args.out_channels)

        elif self.args.model == 'MA-Net':
            self.model = smp.MAnet(
                encoder_name=self.args.encoder,
                encoder_weights=None,
                decoder_use_batchnorm=True, 
                in_channels=self.args.in_channels,
                classes=self.args.out_channels)
        
        elif self.args.model == 'attentionUNet':
            self.model = AttentionUnet(
                spatial_dims=2,
                dropout=0.4,
                in_channels=self.args.in_channels,
                out_channels=self.args.out_channels,
                channels=self.args.attention_channels,
                strides=self.args.attention_strides,
                kernel_size=self.args.attention_kernels,
                up_kernel_size=3)


        elif self.args.model == 'DeepLabV3':
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.args.encoder,
                encoder_weights=None,
                in_channels=self.args.in_channels,
                classes=self.args.out_channels)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=0.3,
                                                               patience=self.args.patience)

        
        scheduler = {"scheduler": scheduler, "step" : "step", "monitor": "Val_Loss" } 
        return [optimizer], [scheduler]
