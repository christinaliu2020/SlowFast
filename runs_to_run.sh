### ucf control runs


# add wget script to get pretrained models

### byol
# training run 50 epochs
python tools/run_net.py --cfg configs/mabe/ucf_BYOL_SlowR50_8x8.yaml DATA.PATH_TO_DATA_DIR /root/ufc/ DATA.PATH_PREFIX /root/ufc/  OUTPUT_DIR /root/ufc_results/byol/ NUM_GPUS 8 BN.NUM_SYNC_DEVICES 8 SOLVER.MAX_EPOCH 50
# eval run 50 epochs
# supervised run 50 epochs
# finetuning from pretrained run 50 epochs
# finetuning eval run from 50 epochs

### mae

# training run 50 epochs
python tools/run_net.py --cfg ../configs/mabe/ucf_VIT_B_16x4_MAE_PT.yaml DATA.PATH_TO_DATA_DIR /root/ufc/ DATA.PATH_PREFIX /root/ufc/ OUTPUT_DIR /root/ufc_results/MAE/ NUM_GPUS 8 SOLVER.MAX_EPOCH 50
# eval run 50 epochs
# supervised run 50 epochs
# finetuning from pretrained run 50 epochs
# finetuning eval run from 50 epochs

### MaskFeat

# training run 50 epochs
python tools/run_net.py --cfg ../configs/mabe/ucf_MVITv2_S_16x4_MaskFeat_PT.yaml DATA.PATH_TO_DATA_DIR /root/ufc/ DATA.PATH_PREFIX /root/ufc/ OUTPUT_DIR /root/ufc_results/mvit/ NUM_GPUS 8 SOLVER.MAX_EPOCH 50
# eval run 50 epochs
# supervised run 50 epochs
# finetuning from pretrained run 50 epochs
# finetuning eval run from 50 epochs
