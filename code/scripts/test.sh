cfg=$1
batch_size=4

pretrained_model='./saved_models/pretrained/pre_coco.pth'
multi_gpus=False
mixed_precision=True

nodes=1
num_workers=1
master_port=11277
stamp=gpu${nodes}MP_${mixed_precision}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=$nodes --master_port=$master_port src/test.py \
                    --stamp $stamp \
                    --cfg $cfg \
                    --mixed_precision $mixed_precision \
                    --batch_size $batch_size \
                    --num_workers $num_workers \
                    --multi_gpus $multi_gpus \
                    --pretrained_model_path $pretrained_model \
