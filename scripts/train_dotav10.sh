export PYTHONPATH=.:$PYTHONPATH

CUDA_VISIBLE_DEVICES=4,5 PORT=6332 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav20.py 2 

CUDA_VISIBLE_DEVICES=4,5 PORT=6332 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav20.py \
    work_dirs/rfcos_ssp_dotav20/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav20

#########################################################################################################

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav20.py 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_test.sh configs/ssp/orcnn_ssp_dotav20.py \
    work_dirs/orcnn_ssp_dotav20/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/orcnn_ssp_dotav20

#########################################################################################################

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_train.sh configs/ssp/redet_ssp_dotav20.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_test.sh configs/ssp/redet_ssp_dotav20.py \
    work_dirs/redet_ssp_dotav20/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/redet_ssp_dotav20