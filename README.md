# VFI-CVD

Code relaese for *Describing Visual Attributes with Pre-trained Image Encoder for Few-shot Fine-grained Recognition*.

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yml
  conda activate VFI-CVD
  ```

## Dataset

Thanks to the contribution of [BiFRN](https://github.com/PRIS-CV/Bi-FRN), we performe our experiments based on their processed few-shot fine-grained recognition datasets:

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]
- dogs \[[Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)\]

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
