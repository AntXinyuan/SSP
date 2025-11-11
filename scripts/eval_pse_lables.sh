export PYTHONPATH=.:$PYTHONPATH

python tools/eval_pseudo_annotations.py \
    --dataset-cfg configs/_base_/datasets/dotav1.py \
    --pse-dir pseudo_labels/release/ssp_dotav10_hybrid/ssp_dotav10_hybrid_2xresolution/
    #--pse-dir pseudo_labels/ssp_dior-new_seed_5e/vor_mix
    #--dataset-cfg configs/_base_/datasets/dior.py \
    #--pse-dir pseudo_labels/rfocs_stage1.5-6e-train/ssp_stage2/