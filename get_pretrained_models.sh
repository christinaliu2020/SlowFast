# get MaskFeat pretrained model
wget https://dl.fbaipublicfiles.com/pyslowfast/masked_models/k400_MVIT_S_MaskFeat_PT_epoch_00300.pyth -P /root/pretrained_models/
# get MAE pretrained model
wget https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth -P /root/pretrained_models/
# get BYOL pretrained model
wget https://dl.fbaipublicfiles.com/pyslowfast/videomoco_models/BYOL_SlowR50_8x8_T2_epoch_00200.pyth -P /root/pretrained_models/