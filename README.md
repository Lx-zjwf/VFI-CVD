# VFI-CVD

Code relaese for *Describing Visual Attributes with Pre-trained Image Encoder for Few-shot Fine-grained Recognition*.

## Overall Framework

![framework](./Figure/overall framework.jpg)

we propose to introduce the vision foundation model as a visual attribute descriptor to facilitate meaningful semantic understanding and promote generalization. A Variational Feature Imitation method Conditioned on Visual Descriptions, namely VFI-CVD, is proposed, which integrates the pre-trained knowledge from a vision foundation model and the expert knowledge captured via a feature extractor, thus generating representations with adequate generalization and fine-grained discrimination capability. To the best of our knowledge, this study is the first to exploit pre-trained knowledge to boost FS-FGR in the meta-learning paradigm.

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate VFI-CVD
  ```

## Dataset

Thanks to the contribution of [BiFRN](https://github.com/PRIS-CV/Bi-FRN), the code is implemeted based on it. We performe experiments based on the processed few-shot fine-grained recognition datasets: [CUB-200-2011](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view), [Stanford Cars](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link), [Stanford Dogs](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link). In addition, we perform experiments on the challenging benchmark dataset iNaturalist2017\[[Data](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz), [Annotation](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)\]. We process iNaturalist into the few-shot version by `./data/init_meta_iNat.py`, which follows [FRN](https://github.com/Tsingularity/FRN). The closed_set studies are conducted on the data processed by `./data/closed_set.py`.

## Train

* To train VFI-CVD on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

```shell
cd experiments/CUB_fewshot_cropped/CVF/Conv-4
./train.sh
```

## Test

```shell
cd experiments/CUB_fewshot_cropped/CVF/Conv-4
python ./test.py
```

## Experiments

The state-of-the-art few-shot fine-grained recognition (FS-FGR) methods compared in our study:
**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:   
:triangular_flag_on_post: | **AAAI** | Cross-Layer and Cross-Sample Feature Optimization Network for Few-Shot Fine-Grained Image Classification | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28208)/[Code](https://github.com/zenith0923/C2-Net)
:triangular_flag_on_post: | **AAAI** | Bi-directional Feature Reconstruction Network for Fine-Grained Few-Shot Image Classification | [Paper](https://arxiv.org/abs/2211.17161)/[Code](https://github.com/PRIS-CV/Bi-FRN)
:scroll: | **TCSVT** | Locally-Enriched Cross-Reconstruction for Few-Shot Fine-Grained Image Classification | [Paper](https://ieeexplore.ieee.org/abstract/document/10123101)/[Code](https://github.com/lutsong/LCCRN)
:triangular_flag_on_post: | **CVPR** | Task Discrepancy Maximization for Fine-Grained Few-Shot Classification | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Task_Discrepancy_Maximization_for_Fine-Grained_Few-Shot_Classification_CVPR_2022_paper.html)/[Code](https://github.com/leesb7426/CVPR2022-Task-Discrepancy-Maximization-for-Fine-grained-Few-Shot-Classification)
