from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch


def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument

    arg("--seed", type=int, default=26012022, help="Random Seed")
    arg("--generator", default=torch.Generator().manual_seed(26012022), help='Train Validate Predict Seed')
    arg("--base_dir1", type=str, default='../input/c2smsfloods/c2smsfloods/c2smsfloods_v1_source_s1/*', help="Sentinel 1 s2c Data Directory")
    arg("--base_dir2", type=str, default='../input/notebookeba979953f/v1.1/data/flood_events', help="Sentinel 1 sen1Floods11 Data Directory")
    arg("--base_dir_source", type=str, default='../input/notebookeba979953f/v1.1/data/flood_events/HandLabeled/S1Hand', help="Sentinel 1 sen1Floods11 source Data Directory")
    arg("--base_dir_label", type=str, default='../input/notebookeba979953f/v1.1/data/flood_events/HandLabeled/LabelHand', help="Sentinel 1 sen1Floods11 label Data Directory")
    arg("--train_path", type=str, default='../input/notebookeba979953f/v1.1/splits/flood_handlabeled/flood_train_data.csv', help="sen1Floods11 Train Data CSV Directory")
    arg("--val_path", type=str, default='../input/notebookeba979953f/v1.1/splits/flood_handlabeled/flood_valid_data.csv', help="sen1Floods11 Val Data CSV Directory")
    arg("--test_path", type=str, default='../input/notebookeba979953f/v1.1/splits/flood_handlabeled/flood_test_data.csv', help="sen1Floods11 Test Data CSV Directory")
    arg("--resize_to", type=tuple, default=(320, 320), help="Shape of the Resized Image")
    arg("--crop_shape", type=int, default=256, help="Shape of the cropped Image")
    arg("--batch_size", type=int, default=20, help="batch size")
    arg("--num_workers", type=int, default=2, help="Number of DataLoader Workers")
    arg("--learning_rate", type=float, default=5e-4, help="Learning Rate")
    arg("--max_lr", type=float, default=1e-2, help="Max Learning Rate")
    arg("--weight_decay", type=float, default=1e-5, help="Weight Decay")
    arg("--in_channels", type=int, default=3, help="Network Input Channels")
    arg("--out_channels", type=int, default=1, help="Network Output Channels")
    arg("--base_features", type=int, default=32, help="Network Channels")

    # UNet, UNet++, AttentionUNet, DeepLabV3 
    arg("--model", type=str, default='UNet', help="SMP model")
    # resnet18, resnet34, resnet50, mobilenet_v2, xception, vgg11_bn, vgg13_bn 
    arg("--encoder", type=str, default='resnet18', help="SMP-TIMM Encoder")
    arg("--attention", type=str, default='scse', help="SMP model")

    arg("--attention_channels", type=list, default=[32, 64, 128, 256], help="AttentionUNet Channels")
    arg("--attention_kernels", type=list, default=[[3, 3]] * 4, help="AttentionUNet Kernels")
    arg("--attention_strides", type=list, default=[[2, 2]] * 3 +  [[1, 1]] , help="AttentionUNet Strides")

    arg("--dynUnet_kernels", type=list, default=[[3, 3]] * 5, help="DynUNet Kernels")
    arg("--dynUnet_strides", type=list, default=[[1, 1]] +  [[2, 2]] * 4, help="DynUNet Strides")
    

    arg("--exec_mode", type=str, default='train', help='Execution Mode')
    arg("--num_epochs", type=int, default=100, help="Number of Epochs")
    arg("--patience", type=int, default=4, help="Patience")    
    arg("--ckpt_path", type=str, default=None, help='Checkpoint Path')
    arg("--save_path", type=str, default='./', help='Saves Path')

    return parser.parse_args(args=[])