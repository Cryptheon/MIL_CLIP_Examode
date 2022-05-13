This CLIP implementation uses Pytorch-Lightning.

# How to Train

python train.py --clip_model_name ViT-B/32 \
  --lmdb_patches_path /projects/0/examode/lmdb/latents/vqvae/experiment_3/ \
  --listed_data_path ./data/cross_validation_folds/10_cross_validation_gt.csv \
  --wsi_to_diagnosis_path ./data/wsi_to_diagnosis_complete.json \
  --clip_config_dir models/configs/ViT.yaml\
  --TransMIL_config_dir models/configs/TransMIL.yaml \
  --image_size 224 \
  --set_size 200 \
  --batch_size 168\
  --train_folds 0 1 2 3 4 5 6 7 8 \
  --gpus 1 \
  --num_workers 2 \
  --num_nodes 1 \
  --shuffle True \
  --accelerator ddp \
  --precision 32 \
  --max_epochs 3500 \
  --check_val_every_n_epoch 20

