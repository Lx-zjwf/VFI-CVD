# VFI-CVD

Code relaese for *Describing Visual Attributes with Pre-trained Image Encoder for Few-shot Fine-grained Recognition*.

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate VFI-CVD
  ```

## Dataset

Thanks to the contribution of [BiFRN](https://github.com/PRIS-CV/Bi-FRN), we performe our experiments based on their processed few-shot fine-grained recognition datasets: [CUB-200-2011](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view), [Stanford Cars](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link), [Stanford Dogs](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link). In addition, we perform experiments on the challenging benchmark dataset iNaturalist2017\[[Data](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_val_images.tar.gz), [Annotation](https://ml-inat-competition-datasets.s3.amazonaws.com/2017/train_2017_bboxes.zip)\]. We process iNaturalist into few-shot version by running `./data/init_meta_iNat.py`, which follows the method in [FRN](https://github.com/Tsingularity/FRN).

**:open_file_folder:** | **Pub.** | **Title** | **Links** 
:-: | :-: | :-  | :-:   
:triangular_flag_on_post: | **AAAI** | Cross-Layer and Cross-Sample Feature Optimization Network for Few-Shot Fine-Grained Image Classification | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28208)/[Code](https://github.com/zenith0923/C2-Net)

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
