export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="data/instance_data"
export CLASS_DIR="data/class_data"
export STYLE_DIR="data/style_data"
export OUTPUT_DIR="output"

CUDA_VISIBLE_DEVICES=0 python lora_diffusion_/magicapture_train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --style_data_dir=$STYLE_DIR \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --with_prior_preservation=False \
  --train_text_encoder \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing\
  --scale_lr \
  --initializer_tokens="person" \
  --learning_rate_unet=1e-4\
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<sks>" \
  --placeholder_tokens_style="<style1>"\
  --save_steps=8000 \
  --max_train_steps_ti=1200 \
  --max_train_steps_tuning=1500 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=4 \
  --log_wandb=False \
  --lambda_arc=0.25 \
  --lambda_style=2 \

 
