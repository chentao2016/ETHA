CUDA_VISIBLE_DEVICES=0 python train.py \
--backbone resnet50 --fold 0 --benchmark pascal --lr 2.8e-4 --bsz 56 --niter 70 --logpath "VOC_RN50_base_cam_0" \
--traincampath ./Datasets_HSN/Base_CAM_VOC_Train/ \
--valcampath ./Datasets_HSN/Base_CAM_VOC_Val/
