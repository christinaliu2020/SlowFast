### ucf control runs

### todos
# adjust memory bank size in all runs

### byol
# training run 50 epochs
python ./tools/run_net.py --cfg configs/mabe/ucf_BYOL_SlowR50_8x8.yaml
# eval run 50 epochs
# supervised run 50 epochs
# finetuning from pretrained run 50 epochs
python ./tools/run_net.py --cfg configs/mabe/finetune_ucf_Slow_R50_syn8.yaml
# finetuning eval run from 50 epochs

### mae

# training run 50 epochs
python ./tools/run_net.py --cfg /configs/mabe/ucf_VIT_B_16x4_MAE_PT.yaml
# eval run 50 epochs
# supervised run 50 epochs
# finetuning from pretrained run 50 epochs
python ./tools/run_net.py --cfg /configs/mabe/finetune_ucf_VIT_B_16x4_MAE_PT.yaml
# finetuning eval run from 50 epochs

### MaskFeat

# training run 50 epochs
python ./tools/run_net.py --cfg /configs/mabe/ucf_MVITv2_S_16x4_MaskFeat_PT.yaml
# eval run 50 epochs
# supervised run 50 epochs
# finetuning from pretrained run 50 epochs
python ./tools/run_net.py --cfg /configs/mabe/finetune_ucf_MVITv2_S_16x4_MaskFeat_PT.yaml
# finetuning eval run from 50 epochs
