export PYTHONPATH=.:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7346 bash ./tools/dist_train.sh \
      configs/ssp/ssp_label_marker.py 4 \
      --seed 2423 \
      --no-validate
