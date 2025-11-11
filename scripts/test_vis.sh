export PYTHONPATH=.:$PYTHONPATH

#CUDA_VISIBLE_DEVICES=4,5 PORT=7345 bash ./tools/dist_test.sh work_dirs/rfcos_ssp_dotav10_adamw_coslr/rfcos_ssp_dotav10_adamw_coslr.py \
#    work_dirs/rfcos_ssp_dotav10_adamw_coslr/epoch_12.pth 2 \
#    --format-only  \
#    --eval-options submission_dir=testmodel/ssp-v2/rfcos_ssp_dotav10_adamw_coslr-12ep

CUDA_VISIBLE_DEVICES=4 PORT=7345 python tools/test.py configs/ssp/rfcos_ssp_dior.py \
    work_dirs/rfcos_ssp_dior/epoch_12.pth \
    --rand_k 30 \
    --show \
    --show-dir vis_results/rfcos_ssp_dior

#CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=7345 bash ./tools/dist_test.sh configs/ssp/ssp_label_marker_v2.py \
#    work_dirs/ssp_label_marker_v2/epoch_1.pth 4 \
#    --eval mAP \
#    --subset val
    #--rand-k 
    #--cfg-options \
    #data.samples_per_gpu=20 \
    #--show \
    #--show-dir vis_results/ssp_label_marker_v2-0911-test-ep12g