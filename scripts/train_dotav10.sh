export PYTHONPATH=/data/liuxinyuan/SSP:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1 PORT=6332 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10.py 2 

CUDA_VISIBLE_DEVICES=0,1 PORT=6332 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav10.py \
    work_dirs/rfcos_ssp_dotav10/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/rfcos_ssp_dotav10

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav10.py 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_test.sh configs/ssp/orcnn_ssp_dotav10.py \
    work_dirs/orcnn_ssp_dotav10/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/orcnn_ssp_dotav10

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=6332 bash ./tools/dist_train.sh configs/ssp/redet_ssp_dotav10.py 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=6332 bash ./tools/dist_test.sh configs/ssp/redet_ssp_dotav10.py \
    work_dirs/redet_ssp_dotav10/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/redet_ssp_dotav10