import cv2
import os
import argparse
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def create_argparser():
    #defaults = dict(dataset_root="./data", det_size=(640, 640))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default="../ffhq_1000/00000")
    parser.add_argument('--det_size', type=tuple_type, default="(512, 512)")
    parser.add_argument('--resize', type=bool, default=False)
    parser.add_argument('--only_face', type=bool, default=False)
    parser.add_argument('--noise_ver', type=bool, default=False)
    return parser


def main():
    args = create_argparser().parse_args()

    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=args.det_size)
    
    try:
        imgs = os.listdir(args.dataset_root)
    except:
        print("There is no directory for --dataset_root")
        exit()
    
    face_dir = args.dataset_root+"_faced"
    os.makedirs(face_dir, exist_ok=True)

    img_dict = dict()
    # dict[img] is consisted of [name, detected_img, [detected_face_list]]
    for x in imgs:
        l = len(x)
        name = x[:l-4] # .jpg
        img = cv2.imread(args.dataset_root+"/"+x)
        mask = np.zeros_like(img)
        faces = app.get(img)
        rimg = app.draw_on(img, faces)
        #cv2.imwrite(face_dir+"/"+name+"_output.jpg", rimg)
        
        img_dict[x] = {'name':name, 'rimg':rimg}

        dimg = img.copy()
        if args.noise_ver:
            noised_image = img.copy()
        detected_face = []
        prev_area = 0
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)

            if box[0] < 0:
                box[0] = 0
            if box[1] < 0:
                box[1] = 0
            if box[2] >= dimg.shape[0]:
                box[2] = dimg.shape[0] - 1
            if box[3] >= dimg.shape[1]:
                box[3] = dimg.shape[1] - 1

            cropped_face = dimg[box[1]:box[3], box[0]:box[2]]
            mask[box[1]:box[3], box[0]:box[2]] = 1
            #print(cropped_face.shape)
            #cv2.imwrite(face_dir+"/"+name+"_face_"+str(i)+".jpg", cropped_face)
            if i == 0: detected_face.append((str(i), cropped_face, cropped_face.shape))
            else:
                if (box[3]-box[1]) * (box[2]-box[0]) > prev_area:
                    detected_face[0] = (str(i), cropped_face, cropped_face.shape)

            prev_area = (box[3]-box[1]) * (box[2]-box[0])
            if args.noise_ver:
                noised_image[box[1]:box[3], box[0]:box[2]] = np.random.randint(0, 256, (box[3]-box[1], box[2]-box[0], 3))
        
        if len(faces) == 0: print(f"***num of detected faces for {x} is {len(faces)}***")
        img_dict[x]['facelist'] = detected_face
        img_dict[x]['faced_img'] = img * mask
        img_dict[x]['masked_img'] = img * (1 - mask)
        if args.noise_ver:
            img_dict[x]['noised'] = noised_image
    
    if args.resize:
        if args.only_face:
            pass
        pass

    else:
        for x in imgs:
            for idx, cropped_face, shape in img_dict[x]['facelist']:
                faceimgname = face_dir+"/"+img_dict[x]['name']+"_face_"+idx+".jpg"
                #cv2.imwrite(faceimgname, cropped_face)

                faceimgname = face_dir+"/"+img_dict[x]['name']+"_faced_"+idx+".jpg"
                cv2.imwrite(faceimgname, img_dict[x]['faced_img'])

                faceimgname = face_dir+"/"+img_dict[x]['name']+"_masked_"+idx+".jpg"
                #cv2.imwrite(faceimgname, img_dict[x]['masked_img'])

            if args.only_face:
                continue
            imgname = face_dir+"/"+img_dict[x]['name']+"_output.jpg"
            #cv2.imwrite(imgname, img_dict[x]['rimg'])

            if args.noise_ver:
                nimgname = face_dir+"/"+img_dict[x]['name']+"_noise.jpg"
                #cv2.imwrite(nimgname, img_dict[x]['noised'])
        

if __name__ == "__main__":
    main()

