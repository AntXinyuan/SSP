cd ..

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_train.sh configs/ssp/redet_ssp_dotav10.py 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_test.sh configs/ssp/redet_ssp_dotav10.py \
    work_dirs/redet_ssp_dotav10/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/redet_ssp_dotav10

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_train.sh configs/ssp/redet_ssp_dotav15.py 4

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_test.sh configs/ssp/redet_ssp_dotav15.py \
    work_dirs/redet_ssp_dotav15/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/redet_ssp_dotav15

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_train.sh configs/ssp/redet_ssp_dotav20.py 4 

CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=6332 bash ./tools/dist_test.sh configs/ssp/redet_ssp_dotav20.py \
    work_dirs/redet_ssp_dotav20/epoch_12.pth 4 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp/redet_ssp_dotav20