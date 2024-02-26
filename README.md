# MagiCapture: High-Resolution Multi-Concept Portrait Customization

[![arXiv](https://img.shields.io/badge/arXiv-2309.06895-b31b1b.svg)](https://arxiv.org/abs/2309.06895)

[[Project Website](https://magicapture.github.io/)]

> **MagiCapture: High-Resolution Multi-Concept Portrait Customization**<br>
> Junha Hyung<sup>1,* </sup>, Jaeyo Shin<sup>2,* </sup>, and Jaegul Choo<sup>1</sup><br>
> <sup>1</sup>KAIST AI, <sup>2</sup>Sogang University
> <sup>*</sup> Equal Contribution

>**Abstract**: <br>
> Large-scale text-to-image models including Stable Diffusion are capable of generating high-fidelity photorealistic portrait images. 
There is an active research area dedicated to personalizing these models, aiming to synthesize specific subjects or styles using provided sets of reference images
However, despite the plausible results from these personalization methods, they tend to produce images that often fall short of realism and are not yet on a commercially viable level. 
This is particularly noticeable in portrait image generation, where any unnatural artifact in human faces is easily discernible due to our inherent human bias. 
To address this, we introduce MagiCapture, a personalization method for integrating subject and style concepts to generate high-resolution portrait images using just a few subject and style references. 
For instance, given a handful of random selfies, our fine-tuned model can generate high-quality portrait images in specific styles, such as passport or profile photos. 
The main challenge with this task is the absence of ground truth for the composed concepts, leading to a reduction in the quality of the final output and an identity shift of the source subject. 
To address these issues, we present a novel Attention Refocusing loss coupled with auxiliary priors, both of which facilitate robust learning within this weakly supervised learning setting. 
Our pipeline also includes additional post-processing steps to ensure the creation of highly realistic outputs. 
MagiCapture outperforms other baselines in both quantitative and qualitative evaluations and can also be generalized to other non-human objects.

## Description
This repo contains the official code for our MagiCapture paper. 

## Updates
**29/01/2024** Sample Code Released!
**09/12/2023** Our paper is accepted to Association for the Advancement of Artificial Intelligence (AAAI), 2024 (23.75% acceptance rate). 

## TODO:
- [x] Release code!
- [ ] Refactoring code, for the convinence.
- [ ] Support SDXL for MagiCapture.

## Setup

```
conda env create -n magicapture
conda activate magicapture
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
cd insightface_/detection/retinaface
make
```

Also, you need to download some models. 

- In [[Model zoo of arcface_torch](https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC)], download glint360k_cosface_r100_fp16_0.1 or ms1mv3_arcface_r100_fp16 to arcface_torch/models directory.
Also, download the [[retinaface](https://drive.google.com/file/d/1_DKgGxQWqlTqe78pw0KavId9BIMNUWfu/view?usp=sharing)] model, and save it in insightface_/detection/retinaface/model


- Download [[this model](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812)] and save it in face_parsing_PyTorch/res/cp


## Usage

### Finetuning

To train the model, run the train_magicapture.sh script.
There is a code as below.
```
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
```

- For instance data, We recommend a picture where the size of the face in the picture is not too big or too small.
- For style data, We recommend a picture that shows the style of the person you want without the size of your face being too small.
- Placeholder tokens are enough to one token per concept.
- In the paper, we use 1.2k ti, 1.5k tuning steps. 1.6k for ti, 2.0k for tuning step with lr_unet=4e-5, lr_text=5e-6 is also fine.  
However, some data can converge much faster, or much slower. The best setting of parameters can vary depending on the given training data, so we recommend you try many things.
- In lora_diffusion_/loss_utils.py, you can train by choosing a face recognition model between cosface and arcface. Both of them were fine.
- lambda_arc is the weight applied to the id loss (arcface loss). 0.1 ~ 0.5 was fine, but it is data-dependent.
- lambda_style is the weight applied to the style mse loss, at the first half of training (both ti and unet). Due to the id loss, which is cosine similarity between face embeddings, the object loss(instance mse + id loss) is much bigger than style mse loss, until the id loss stabilizes. Thus we utilize the weighted style loss, to adjust the balance between object and style loss value.

### Inference

To generate new images for the personality and style, run the inference.sh script.
There is a code as below.
```
CUDA_VISIBLE_DEVICES=0 python inference_magicapture.py --lora_path output --pretrained_model_name runwayml/stable-diffusion-v1-5 --step 50 --output_folder_prefix outputtest --prompt "A photo of a <sks> person with style <style1>" --seed 42

```


## Tips and Tricks
- We use the pre-defined template, such as "A photo of a <sks> person". It performences better than "A phofo of a <sks>".
- Results can be seed sensititve, data sensitive. If you're unsatisfied with the model, try re-inverting with a new seed, or use negative prmompt. Otherwise, try re-training with different (but similar) arguments. Learning rate and lambda_arc can be adjusted for this.


## Stable Diffusion XL

Stable Diffusion XL support is a work in progress and will be completed soon™.


## Citation

If you make use of our work, please cite our paper:

```
@article{hyung2023magicapture,
  title={Magicapture: High-resolution multi-concept portrait customization},
  author={Hyung, Junha and Shin, Jaeyo and Choo, Jaegul},
  journal={arXiv preprint arXiv:2309.06895},
  year={2023}
}
```

## Results
Please visit our [project page](https://magicapture.github.io/) or read our paper for more!

## Acknowledgement
This project code is based on [[this repository](https://github.com/cloneofsimo/lora)]. Also, some codes are brought from [[Codeformer](https://github.com/sczhou/CodeFormer)], [[insightface](https://github.com/deepinsight/insightface)], [[face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)]. Thanks for their awesome works.
