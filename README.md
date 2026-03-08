## 📄 ITSELF: Attention Guided Fine-Grained Alignment for Vision–Language Retrieval (WACV 2026 (Oral))

**Tien-Huy Nguyen<sup>1,2</sup>, Huu-Loc Tran<sup>1,2</sup>, Thanh Duc Ngo<sup>1,2</sup>**

<sup>1</sup> University of Information Technology  
<sup>2</sup> Vietnam National University - HCM  

<p align="left">
  <a href="https://arxiv.org/pdf/2601.01024v1"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/GenAI4ELab"><img src="https://img.shields.io/badge/🤗-model-yellow" alt="HuggingFace Checkpoint"></a>
  <a href="https://trhuuloc.github.io/itself/"><img src="https://img.shields.io/badge/ProjectPage-up-green" alt="Project Page"></a>
  <a href="https://drive.google.com/file/d/1H6QSuwBRGqEni7Zk1ZIaBk-iv0BTC2cQ/view?usp=share_link"><img src="https://img.shields.io/badge/Poster-up-blue" alt="Poster"></a>
  <a href="https://www.youtube.com/watch?v=afy9iBuSDM8"><img src="https://img.shields.io/badge/Video-up-pink" alt="Project Page"></a>
  
</p>

## News
- [03/08/2026] Check out our checkpoints, now available on [[Link](https://huggingface.co/GenAI4ELab)]. 

- [01/22/2026] Our paper has been accepted for an oral presentation (Acceptance rate ~3.5%).

- [11/11/2025] Our paper has been accepted at WACV 2026.


## Highlights
- We propose ITSELF framework. A novel attention-guided implicit local alignment framework, ITSELF with GRAB that leverages encoder attention to mine fine-grained discriminativecues and reinforce global text-image alignment without additional supervision.
- Robust Selection & Scheduling. We propose MARS, which fuses attention across layers and performs diversity-aware top-k selection; and ATS, which anneals the retention budget from coarse to fine over training to stabilize learning and prevent early information loss.

![ITSELF Framework](framework.png)


## Citation 
If you find this repository useful, please use the following BibTeX entry for citation.

```
@misc{nguyen2026itselfattentionguidedfinegrained,
      title={ITSELF: Attention Guided Fine-Grained Alignment for Vision-Language Retrieval}, 
      author={Tien-Huy Nguyen and Huu-Loc Tran and Thanh Duc Ngo},
      year={2026},
      eprint={2601.01024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.01024}, 
}
```

## License

This project is released under the MIT license. 
