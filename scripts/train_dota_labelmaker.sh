export PYTHONPATH=.:$PYTHONPATH

#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7355 bash ./tools/dist_train.sh \
#      configs/ssp/ssp_label_marker-0.1delta.py 4 \
#      --no-validate
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7355 bash ./tools/dist_train.sh \
#      configs/ssp/ssp_label_marker-0.2delta.py 4 \
#      --no-validate
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7355 bash ./tools/dist_train.sh \
#      configs/ssp/ssp_label_marker-0.3delta.py 4 \
#      --no-validate
#
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_train.sh \
#      configs/ssp/ssp_label_marker-0.1delta.py 4 \
#      --checkpoint work_dirs/ssp_label_marker-0.1delta/epoch_5.pth \
#      --seed 2423 \
#      --no-validate \
#      --cfg-options \
#      data.samples_per_gpu=10\
#      optimizer.lr=0.0 \
#      runner.max_epochs=1 \
#      log_config.interval=1 \
#      checkpoint_config.interval=-1 \
#      checkpoint_config.save_last=False \
#      model.bbox_head.is_record_stage=True \
#      model.train_cfg.pseudo_label_dir=pseudo_labels/ssp_dotav10_5e_0.1delta
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_train.sh \
#      configs/ssp/ssp_label_marker-0.2delta.py 4 \
#      --checkpoint work_dirs/ssp_label_marker-0.2delta/epoch_5.pth \
#      --seed 2423 \
#      --no-validate \
#      --cfg-options \
#      data.samples_per_gpu=10\
#      optimizer.lr=0.0 \
#      runner.max_epochs=1 \
#      log_config.interval=1 \
#      checkpoint_config.interval=-1 \
#      checkpoint_config.save_last=False \
#      model.bbox_head.is_record_stage=True \
#      model.train_cfg.pseudo_label_dir=pseudo_labels/ssp_dotav10_5e_0.2delta
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_train.sh \
#      configs/ssp/ssp_label_marker-0.3delta.py 4 \
#      --checkpoint work_dirs/ssp_label_marker-0.3delta/epoch_5.pth \
#      --seed 2423 \
#      --no-validate \
#      --cfg-options \
#      data.samples_per_gpu=10\
#      optimizer.lr=0.0 \
#      runner.max_epochs=1 \
#      log_config.interval=1 \
#      checkpoint_config.interval=-1 \
#      checkpoint_config.save_last=False \
#      model.bbox_head.is_record_stage=True \
#      model.train_cfg.pseudo_label_dir=pseudo_labels/ssp_dotav10_5e_0.3delta

CUDA_VISIBLE_DEVICES=2,3 PORT=7355 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10-0.1delta.py 2

CUDA_VISIBLE_DEVICES=2,3 PORT=7355 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10-0.2delta.py 2

CUDA_VISIBLE_DEVICES=2,3 PORT=7355 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10-0.3delta.py 2

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav10-0.1delta.py \
    work_dirs/rfcos_ssp_dotav10-0.1delta/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav10-0.1delta

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav10-0.2delta.py \
    work_dirs/rfcos_ssp_dotav10-0.2delta/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav10-0.2delta

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav10-0.3delta.py \
    work_dirs/rfcos_ssp_dotav10-0.3delta/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav10-0.3delta
