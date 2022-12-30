import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataloader import *
from args import get_main_args
from model import *
import albumentations as A
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer


if __name__ == "__main__":
    args = get_main_args()
    transformations = A.Compose([A.RandomCrop(*args.resize_to),
                                A.RandomRotate90(),
                                A.HorizontalFlip(),
                                A.VerticalFlip()])


    dm = FloodDataModule(args, transformations)
    dm.setup()
    model = Unet(args)
    

    model_ckpt = ModelCheckpoint(dirpath="./",
                                filename="best_model",
                                monitor="dice_mean",
                                mode="max",
                                save_last=True)


    early_stop_callback = EarlyStopping(
        monitor="dice_mean",
        patience=(args.patience * 3),
        mode="max")

    callbacks = [model_ckpt, early_stop_callback]


    trainer = Trainer(callbacks=callbacks,
                    enable_checkpointing=True,
                    max_epochs=args.num_epochs, 
                    enable_progress_bar=True,
                    log_every_n_steps=len(dm.train_dataloader()),
                    devices=1,
                    accelerator="gpu",
                    amp_backend='apex',
                    profiler='simple',
                    detect_anomaly=True)


    # train the model
    if args.exec_mode == 'train':
        trainer.fit(model, dm)
    else:
        trainer.predict(model, datamodule=dm, ckpt_path=args.ckpt_path) 
        
