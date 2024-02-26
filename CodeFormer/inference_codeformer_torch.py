import os
import cv2
import argparse
import glob
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

import sys
#print(os.path.join(os.path.dirname(__file__), './basicsr'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), './basicsr'))

from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper_torch import FaceRestoreHelper # 바로 이게 문제
from facelib.utils.misc import is_gray
from basicsr.utils import imwrite, img2tensor, tensor2img

from basicsr.utils.registry import ARCH_REGISTRY

from PIL import Image



#sys.path.insert(0, os.getcwd())

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan(bg_tile):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

def Codeformer_tensor(
    input_path='./inputs/whole_imgs',
    input_tensor=None, # maybe input image, decoded latent, C x H x W, clamped in (0, 1)
    output_path=None,
    fidelity_weight=0.5,
    upscale=1,
    has_aligned=False,
    only_center_face=False,
    draw_box=False,
    detection_model='retinaface_resnet50',
    bg_upsampler="None",
    face_upsample="False",
    bg_tile=400,
    suffix=None,
    save_video_fps=None,
    net=None,
    face_helper=None,
):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()

    # ------------------------ input & output ------------------------
    w = fidelity_weight
    input_video = False
    if input_tensor is not None:
        input_img_list = [input_tensor]
    elif input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [input_path]
        result_root = f'results/test_img_{w}'
    else: # input img folder
        if input_path.endswith('/'):  # solve when path ends with /
            input_path = input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(input_path)}_{w}'

    if not output_path is None: # set output path
        result_root = output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # TODO : bg upsampler, face upsampler 관련 코드도 전부 수정해주어야할듯 ==> X. 이건 할 필요 없음
    # ------------------ set up background upsampler ------------------
    if bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan(bg_tile)
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(bg_tile)
    else:
        face_upsampler = None


    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    #if not has_aligned: 
        #print(f'Face detection model: {detection_model}')
    #if bg_upsampler is not None: 
        #print(f'Background upsampling: True, Face upsampling: {face_upsample}')
    #else:
        #print(f'Background upsampling: False, Face upsampling: {face_upsample}')


    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    # edited version : for tensored image
    #output = []
    for i, img in enumerate(input_img_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
            
        #img_name = os.path.basename(img_path)
        #basename, ext = os.path.splitext(img_name)
        #print(f'[{i+1}/{test_img_num}] Processing: Tensor')
        #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        

        if has_aligned: 
            # if the input faces are already cropped and aligned
            if img.shape[2] != 512: img = F.interploate(img, 512, mode=linear)
            face_helper.is_gray = is_gray(img, threshold=10)
            #if face_helper.is_gray:
                #print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            # face_helper.read_image(img)
            face_helper.input_img = (img * 255).flip(dims=(0,)) # then, img is in (0, 255), and BGR..? 이건 확인 필요할듯 같은 결과가 나오는지...
            face_helper.input_img_numpy = cv2.cvtColor((img * 255).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
            # cv2.imwrite('test_cv2img.jpg', face_helper.input_img_numpy)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            #print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        if len(face_helper.cropped_faces) == 0:
            print(f"Face is not detected by det model: {detection_model}, at codeformer")
            return None, None, None, None
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            #cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            cropped_face_t = cropped_face / 255.
            cropped_face_t = cropped_face_t.flip(dims=(0,)) # now rgb
            return_cropped_face = torch.empty_like(cropped_face_t).copy_(cropped_face_t).unsqueeze(0) # clamped in (0, 1), C x H x W

            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                out = net(cropped_face_t, w=w, adain=True) # ==> this is the... target of mse?
                output = out[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                output = (output / 2 + 0.5).clamp(0, 1) # clamped in 0, 1, C x H x W
                del output
                torch.cuda.empty_cache() # 메모리가 얼마나 드는 지를 모르겠네... 많이 들면 gpu로 부족할수도
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                return None, None, None, None
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

    restored_img = img2tensor(restored_img / 255., bgr2rgb=True, float32=True).unsqueeze(0).permute(0, 2, 3, 1)
    return restored_img

    #print(f'\nAll results are saved in {result_root}')

def get_net_and_face_helper():
    device = get_device()
    codeformer_net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                        connect_list=['32', '64', '128', '256']).to(device)

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()

    face_helper = FaceRestoreHelper(
    1,
    face_size=512,
    crop_ratio=(1, 1),
    det_model = 'retinaface_resnet50',
    save_ext='png',
    use_parse=True,
    device=device)

    return codeformer_net, face_helper


if __name__ == '__main__':
    img = cv2.imread('../faceguide_images_tests/RV2-anya-9img-7-3-nf-semi-arc-200_1_feature/_prompt_3_dpm_negatived_lam_20.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).permute(2, 0, 1) / 255.


    aligned_face, output, _, _ = Codeformer_tensor(
        input_path=None,
        input_tensor=img, # maybe input image, decoded latent, C x H x W, clamped in (0, 1)
        output_path=None,
        fidelity_weight=0.5,
        upscale=1,
        has_aligned=False,
        only_center_face=False,
        draw_box=False,
        detection_model='retinaface_resnet50',
        bg_upsampler="None",
        face_upsample="False",
        bg_tile=400,
        suffix=None,
        save_video_fps=None,
    )

    output = output[0] * 255
    output = output.permute(1, 2, 0).detach().cpu().numpy()
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite('testfortorch.jpg', output)
