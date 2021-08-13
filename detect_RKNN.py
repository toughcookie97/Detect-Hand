from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
#from DTW import judgeGesture
from rknn.api import RKNN
parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.2, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}




def test_pictures():
    c=0
    resize = 1
    fr=0
    path_root='demo/images/'
    path_root_save='demo/results/'
    img_list=os.listdir(path_root)
    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform=['rv1126'])
    print('done')

    ret = rknn.load_rknn('./model811133.rknn')
   # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Build model



    if ret != 0:
        print('load model failed!')
        exit(ret)
    print('done')

    for image_name in img_list:
        image_path = os.path.join(path_root,image_name)#"./curve/test.jpg"
        img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_raw=cv2.resize(img_raw,(320,320))

        img = np.float32(img_raw)

        im_height, im_width= img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= 0.299 * 104 + 0.587 * 117 + 0.114 * 123
        #img = img.unsqueeze(0)#transpose(2, 0, 1)

        print("img size")
        print(img_raw.shape)
        tic = time.time()
        print(img.shape)
        #loc, conf = net(img)  # forward pass
        loc,conf=rknn.inference(inputs=[img])
        loc=torch.Tensor(loc)
        conf=torch.Tensor(conf)
        print(loc.shape)
        print(conf.shape)
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward()

        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        '''
        scale1 = torch.Tensor([img_raw.shape[3], img_raw.shape[2], img_raw.shape[3], img_raw.shape[2],
                               img_raw.shape[3], img_raw.shape[2], img_raw.shape[3], img_raw.shape[2],
                               img_raw.shape[3], img_raw.shape[2]])
        scale1 = scale1.to(device)
        '''
        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        #dets = np.concatenate((dets, landms), axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                #cv2.putText(img_raw, text, (cx, cy),
                #            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                text1="({},{})".format((b[2]+b[0])/2,(b[3]+b[1])/2)
               
                cv2.putText(img_raw, text1, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                #
            # save image

            #
            path_save=os.path.join(path_root_save,image_name)
            cv2.imwrite(path_save, img_raw)

def test_video():
    cap=cv2.VideoCapture(0)
    frame_gestures=[]
    ff=0
    while 1 :
        resize=1
        
        ret, frame = cap.read()
        frame=cv2.resize(frame,(320,320))
        ff=ff+1
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
        img_raw = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        img_raw=cv2.resize(img_raw,(320,320))
        img = np.float32(img_raw)
        im_height, im_width= img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= 0.299 * 104 + 0.587 * 117 + 0.114 * 123
        #img = img.unsqueeze(0)#transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf = net(img)  # forward pass
        #输出在网络推理中花费的时间
        #print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]

        #dets = np.concatenate((dets, landms), axis=1)

        # show image
      
        for b in dets:
                if b[4] < args.vis_thres:
                    continue
                #text = "{:.4f}".format(b[4])
                #text=""
                b = list(map(int, b))
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                text1="({},{})".format((b[2]+b[0])/2,(b[3]+b[1])/2)
                #cv2.putText(frame, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.putText(frame, text1, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                #if ff%2==0:
                frame_gestures.append([(b[2]+b[0])/2,(b[3]+b[1])/2])
                #
            # save image

            #
        cv2.imshow("capture", frame)
            
    
    cap.release()
    cv2.destroyAllWindows()
    return frame_gestures
if __name__ == '__main__':
    
    test_pictures()
    #frameges=[]
    #frameges=test_video()
    #judgeGesture(frameges)
    
        
        

        

        
        
    
