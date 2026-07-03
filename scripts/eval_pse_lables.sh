export PYTHONPATH=.:$PYTHONPATH

python tools/eval_pseudo_annotations.py \
    --dataset-cfg configs/_base_/datasets/dotav1.py \
    --pse-dir pseudo_labels/release/ssp_dotav10_hybrid/ssp_dotav10_hybrid_2xresolution/
