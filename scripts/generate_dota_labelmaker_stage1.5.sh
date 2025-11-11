export PYTHONPATH=.:$PYTHONPATH
#--checkpoint work_dirs/ssp_label_marker_v2/epoch_12.pth \
CUDA_VISIBLE_DEVICES=6,7 PORT=7345 bash ./tools/dist_train.sh \
      configs/ssp/rfcos_ssp_dotav10_adamw_coslr.py 2 \
      --checkpoint work_dirs/rfcos_ssp_dotav10_adamw_coslr_6e/epoch_6.pth \
      --no-validate \
      --cfg-options \
      data.samples_per_gpu=12 \
      optimizer.lr=0.0 \
      runner.max_epochs=1 \
      log_config.interval=1 \
      checkpoint_config.interval=-1 \
      checkpoint_config.save_last=False \
      model.bbox_head.is_record_stage=True \
      model.train_cfg.pseudo_label_dir=pseudo_labels/rfocs_stage1.5-6e-train
      #--seed 2423 \