export PYTHONPATH=.:$PYTHONPATH

#CUDA_VISIBLE_DEVICES=4,5 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10_0906a.py 2 

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10.py 2 

CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav10.py \
    work_dirs/rfcos_ssp_dotav10/epoch_12.pth 2 \
    --format-only  \
    --eval-options submission_dir=testmodel/ssp-topk/rfcos_ssp_dotav10
#
#CUDA_VISIBLE_DEVICES=7,8 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav15.py 2
##
#CUDA_VISIBLE_DEVICES=0,1 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav15.py \
#    work_dirs/rfcos_ssp_dotav15/epoch_12.pth 2 \
#    --format-only  \
#    --eval-options submission_dir=testmodel/ssp/rfcos_ssp_dotav15
#
#CUDA_VISIBLE_DEVICES=0,1 PORT=2344 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav20.py 2 
#
#CUDA_VISIBLE_DEVICES=0,1 PORT=2344 bash ./tools/dist_test.sh configs/ssp/rfcos_ssp_dotav20.py \
#    work_dirs/rfcos_ssp_dotav20/epoch_12.pth 2 \
#    --format-only  \
#    --eval-options submission_dir=testmodel/ssp/rfcos_ssp_dotav20


#PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=6,7 PORT=2555 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10_512ch_adamw_coslr.py 2
#PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=4,5 PORT=2515 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10_adamw_coslr_stage2.py 2
#
#PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=6,7 PORT=2555 bash ./tools/dist_train.sh configs/ssp/rfcos_ssp_dotav10_adamw_coslr_6e.py 2
#
#PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_train.sh configs/ssp/orcnn_ssp_dotav10.py 4 
#
#PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=2,3 PORT=2344 bash ./tools/dist_train.sh configs/ssp/ssp_label_marker-speed.py 2 