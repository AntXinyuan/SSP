export PYTHONPATH=.:$PYTHONPATH

#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=5222 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav15.py 4 
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=5322 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav15.py \
#    work_dirs/rfcos_ssp_dotav15/epoch_12.pth 4 \
#    --format-only  \
#    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav15

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=5322 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav15.py 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=5322 bash ./tools/dist_test.sh configs/ssp/orcnn_ssp_dotav15.py \
    work_dirs/orcnn_ssp_dotav15/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/orcnn_ssp_dotav15

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=5322 bash ./tools/dist_train.sh configs/ssp/redet_ssp_dotav15.py 4

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=5322 bash ./tools/dist_test.sh configs/ssp/redet_ssp_dotav15.py \
    work_dirs/redet_ssp_dotav15/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/redet_ssp_dotav15