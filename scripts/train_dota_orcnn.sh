cd ..
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav10.py 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_test.sh configs/ssp/orcnn_ssp_dotav10.py \
    work_dirs/orcnn_ssp_dotav10/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/orcnn_ssp_dotav10

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav15.py 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_test.sh configs/ssp/orcnn_ssp_dotav15.py \
    work_dirs/orcnn_ssp_dotav15/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/orcnn_ssp_dotav15

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav20.py 4 

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=7345 bash ./tools/dist_test.sh configs/ssp/orcnn_ssp_dotav20.py \
    work_dirs/orcnn_ssp_dotav20/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/orcnn_ssp_dotav20