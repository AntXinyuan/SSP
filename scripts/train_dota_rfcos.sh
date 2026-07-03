export PYTHONPATH=.:$PYTHONPATH

#CUDA_VISIBLE_DEVICES=4,5 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10_0906a.py 2 

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10.py 2 

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav10.py \
    work_dirs/rfcos_ssp_dotav10/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav10

#########################################################################################################

CUDA_VISIBLE_DEVICES=7,8 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav15.py 2

CUDA_VISIBLE_DEVICES=0,1 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav15.py \
    work_dirs/rfcos_ssp_dotav15/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/rfcos_ssp_dotav15

#########################################################################################################

CUDA_VISIBLE_DEVICES=0,1 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav20.py 2 

CUDA_VISIBLE_DEVICES=0,1 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav20.py \
    work_dirs/rfcos_ssp_dotav20/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/rfcos_ssp_dotav20
