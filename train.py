import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CLIPWrapper

def main(hparams):
    clip_config_dir = hparams.clip_config_dir
    transMIL_config_dir = hparams.TransMIL_config_dir

    with open(clip_config_dir) as fin:
        clip_config = yaml.safe_load(fin)[hparams.clip_model_name]

    with open(transMIL_config_dir) as fin:
        transmil_config = yaml.safe_load(fin)["TransMIL"]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    print("clip config: ", clip_config)
    print("TransMIL config: ", transmil_config)
    print("hparams: ", hparams)
    dm_train = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, log_every_n_steps=2)
    model = CLIPWrapper(clip_config, transmil_config, hparams)
    trainer.fit(model, train_dataloaders=dm_train)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--clip_config_dir', type=str, default='models/configs/ViT.yaml')
    parser.add_argument('--TransMIL_config_dir', type=str, default='models/configs/TransMIL.yaml')
    parser.add_argument('--vision_lr', type=float, default=8e-5)
    parser.add_argument('--lm_lr', type=float, default=1e-5)
    parser.add_argument('--restart_lr_after', type=int, default=500)
    parser.add_argument('--clip_model_name', type=str, default='ViT-B/32')
    parser.add_argument('--val_carcinoma_txts', type=str, default='data/validation_txts/carcinoma/carcinoma_reports_fold_9.txt')

    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
