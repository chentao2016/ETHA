# Multimodal Hierarchical Attention Framework for Efficient Weakly Supervised Few-Shot Segmentation under SAGIN Environment

## Reqirements

```
# create conda env
conda create -n etha python=3.9
conda activate etha

# install packages
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python ftfy regex tqdm ttach tensorboard lxml cython

# install pydensecrf from source
git clone https://github.com/lucasb-eyer/pydensecrf
cd pydensecrf
python setup.py install
```

## Preparing Few-Shot Segmentation Datasets
Download following datasets:

> #### 1. PASCAL-5<sup>i</sup>
> Download PASCAL VOC2012 devkit (train/val data):
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from HSNet [[Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)].

> #### 2. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations:
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```
> Download COCO2014 train/val annotations from HSNet Google Drive: [[train2014.zip](https://drive.google.com/file/d/1cwup51kcr4m7v9jO14ArpxKMA4O3-Uge/view?usp=sharing)], [[val2014.zip](https://drive.google.com/file/d/1PNw4U3T2MhzAEBWGGgceXvYU3cZ7mJL1/view?usp=sharing)].
> (and locate both train2014/ and val2014/ under annotations/ directory).



Create a directory '../Datasets_HSN' for the above three few-shot segmentation datasets and appropriately place each dataset to have following directory structure:

    ../                         # parent directory
    ├── ./                      # current (project) directory
    │   ├── common/             # (dir.) helper functions
    │   ├── data/               # (dir.) dataloaders and splits for each FSSS dataset
    │   ├── model/              # (dir.) implementation of Hypercorrelation Squeeze Network model 
    │   ├── README.md           # intstruction for reproduction
    │   ├── train.py            # code for training ETHA
    │   └── test.py             # code for testing ETHA
    └── Datasets_HSN/
        ├── VOC2012/            # PASCAL VOC2012 devkit
        │   ├── Annotations/
        │   ├── ImageSets/
        │   ├── ...
        │   └── SegmentationClassAug/
        ├── COCO2014/           
        │   ├── annotations/
        │   │   ├── train2014/  # (dir.) training masks (from Google Drive) 
        │   │   ├── val2014/    # (dir.) validation masks (from Google Drive)
        │   │   └── ..some json files..
        │   ├── train2014/
        │   └── val2014/
        ├── CAM_VOC_Train/ 
        ├── CAM_VOC_Val/ 
        └── CAM_COCO/

## Preparing pre-trained model
Download CLIP pre-trained [ViT-B/16] at [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt),
ImageNet pre-trained [ResNet-50] at [here](https://download.pytorch.org/models/resnet50-19c8e357.pth), ImageNet pre-trained [ResNet-101] at [here](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth),
and put them to `./pretrain/`.

## Preparing CAM for Few-Shot Segmentation Datasets
Generate base CAM following IMR-HSNet:
> #### 1. PASCAL-5<sup>i</sup> base CAM
> ```bash
> python generate_base_cam_voc.py --traincampath ../Datasets_HSN/Base_CAM_VOC_Train/
>                                 --valcampath ../Datasets_HSN/Base_CAM_VOC_Val/
> ```
>#### 2. COCO-20<sup>i</sup> base CAM
> ```bash
> python generate_base_cam_coco.py --campath ../Datasets_HSN/Base_CAM_COCO/
generate refined CAM following CLIP-ES:
>#### 3. PASCAL-5<sup>i</sup> refined CAM
> ```bash
> python generate_refined_cams_voc.py --split_file ./pytorch_grad_cam_refined/voc12/train_aug.txt
> python generate_refined_cams_voc.py --split_file ./pytorch_grad_cam_refined/voc12/val.txt
> ```
>#### 4. COCO-20<sup>i</sup> refined CAM
> ```bash
> python generate_refined_cams_coco.py --split_file ./pytorch_grad_cam_refined/coco14/train.txt
> python generate_refined_cams_coco.py --split_file ./pytorch_grad_cam_refined/coco14/val.txt
> ```
> 
## Training
> #### 1. train on PASCAL-5<sup>i</sup> with base CAM
> ```bash
> ./train_base.sh

> #### 2. train on COCO-20<sup>i</sup> with base CAM
> ```bash
> ./train_base_coco.sh
change `data/dataset.py` line 18-19:
>'pascal': DatasetPASCAL,
> 
>'coco': DatasetCOCO,

> #### 3. train on PASCAL-5<sup>i</sup> with refined CAM
> ```bash
> ./train_refined.sh

> #### 4. train on COCO-20<sup>i</sup> with refined CAM
> ```bash
> ./train_refined_coco.sh

## Testing
> ```bash
> python test.py --backbone {resnet50,resnet101} 
>                --fold {0, 1, 2, 3} 
>                --benchmark pascal
>                --nshot {1, 5} 
>                --load "path_to_trained_model/best_model.pt"
> ```
