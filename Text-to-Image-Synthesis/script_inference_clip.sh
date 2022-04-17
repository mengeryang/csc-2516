# python3 runtime_clip.py --dataset birds --inference \
# --pre_trained_disc checkpoints_clip/disc_190.pth \
# --pre_trained_gen checkpoints_clip/gen_190.pth \
# --split 2 \
# --save_path clip_190ep_scratch
# python3 runtime_clip.py --dataset birds --inference \
# --pre_trained_disc checkpoints_clip_resume/disc_130.pth \
# --pre_trained_gen checkpoints_clip_resume/gen_130.pth \
# --split 2 \
# --save_path clip_190ep_resume
python3 runtime_bert.py --dataset birds --inference \
--pre_trained_disc checkpoints_bert/disc_190.pth \
--pre_trained_gen checkpoints_bert/gen_190.pth \
--split 2 \
--save_path bert_190ep
