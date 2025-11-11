export PYTHONPATH=.:$PYTHONPATH
#--checkpoint work_dirs/ssp_label_marker_v2/epoch_12.pth \
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_train.sh \
      configs/ssp/ssp_label_marker_dior.py 4 \
      --checkpoint work_dirs/ssp_label_marker_dior/epoch_5.pth \
      --no-validate \
      --cfg-options \
      data.samples_per_gpu=16\
      optimizer.lr=0.0 \
      runner.max_epochs=1 \
      log_config.interval=1 \
      checkpoint_config.interval=-1 \
      checkpoint_config.save_last=False \
      model.bbox_head.is_record_stage=True \
      model.train_cfg.pseudo_label_dir=pseudo_labels/ssp_dior-new_seed_new_cfg_5e