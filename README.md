# Semantic-decoupled Spatial Partition Guided Point-supervised Oriented Object Detection

[![arxiv](https://img.shields.io/badge/arXiv-2506.10601-479ee2.svg)](https://arxiv.org/pdf/2506.10601)
[![TopoPoint](https://img.shields.io/badge/GitHub-SSP-blueviolet.svg)](https://github.com/antxinyuan/ssp)

🔥 We appreciate the attention to our paper. The code will be organized and released as soon as possible.

> Production from Institute of Computing Technology, Chinese Academy of Sciences.

> Primary contact: Xinyuan Liu ( liuxinyuan21s@ict.ac.cn ).

## TL;DR
This repository contains the source code of [**Semantic-decoupled Spatial Partition Guided Point-supervised Oriented Object Detection**](https://arxiv.org/pdf/2506.10601).

To tackle inadequate sample assignment and instance confusion in point-supervised oriented object detection for remote sensing dense scenes, we propose SSP (Semantic-decoupled Spatial Partition), a framework integrating rule-driven prior injection and data-driven label purification. Its core innovations include pixel-level spatial partition for sample assignment and semantic-modulated box extraction for pseudo-label generation. 

![method](figs/pipeline.png "Method pipeline")

## Updates 
- [2025.6.13] TopoPoint paper is released at [arXiv](https://arxiv.org/abs/2506.10601).

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

``` bibtex
@misc{liu2025ssp,
      title={Semantic-decoupled Spatial Partition Guided Point-supervised Oriented Object Detection}, 
      author={Xinyuan Liu and Hang Xu and Yike Ma and Yucheng Zhang and Feng Dai},
      year={2025},
      eprint={2506.10601},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.10601}, 
}

@inproceedings{xu2024acm,
  title={Rethinking boundary discontinuity problem for oriented object detection},
  author={Xu, Hang and Liu, Xinyuan and Xu, Haonan and Ma, Yike and Zhu, Zunjie and Yan, Chenggang and Dai, Feng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17406--17415},
  year={2024}
}

```

## Related resources

We acknowledge all the open-source contributors for the following projects to make this work possible:
- [PointOBB-v2](https://github.com/VisionXLab/PointOBB-v2)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
