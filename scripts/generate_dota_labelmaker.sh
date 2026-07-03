export PYTHONPATH=.:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_train.sh \
      configs/ssp/ssp_label_marker.py 4 \
      --checkpoint work_dirs/ssp_label_marker/epoch_5.pth \
      --seed 2423 \
      --no-validate \
      --cfg-options \
      data.samples_per_gpu=10\
      optimizer.lr=0.0 \
      runner.max_epochs=1 \
      log_config.interval=1 \
      checkpoint_config.interval=-1 \
      checkpoint_config.save_last=False \
      model.bbox_head.is_record_stage=True \
      model.train_cfg.pseudo_label_dir=pseudo_labels/ssp_dotav10_hybrid