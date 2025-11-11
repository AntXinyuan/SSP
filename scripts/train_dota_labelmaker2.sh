export PYTHONPATH=.:$PYTHONPATH

CUDA_VISIBLE_DEVICES=6,7 PORT=7564 bash ./tools/dist_train.sh \
      configs/ssp/ssp_label_marker-speed.py 2 \
      --no-validate
