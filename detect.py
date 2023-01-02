# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import pytesseract
from datetime import datetime, timedelta, time
import math
import csv
from collections import deque
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


# movement of crickets in the video (points)
pts = deque()

raw_data = {'frame': [],
    'eating_crickets': [],
    'drinking_crickets': [],
    'outside_crickets': []}

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h+20), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)


def drawInterface(im0,
                  feeder_x,
                  feeder_y,
                  feeder_w,
                  feeder_h,
                  water_x,
                  water_y,
                  water_w,
                  water_h,
                  eating_crickets,
                  watering_crickets,
                  outside_cirkcets,
                  feeder_r,
                  ):
    # feeder
    # cv2.rectangle(im0, (feeder_x, feeder_y, feeder_w, feeder_h), (0, 0, 255), 2)
    cv2.circle(im0, (feeder_x, feeder_y), feeder_r, (0, 0, 255), 2)

    # water dispenser
    cv2.rectangle(im0, (water_x, water_y, water_w, water_h), (255, 0, 0), 2)

    # Mos: print the number of detected crickets
    # print("Eating: ", eating_crickets)
    # print("Watering: ", watering_crickets)
    # print("Outside: ", outside_cirkcets)
    draw_text(im0, "Eating: " + str(eating_crickets), cv2.FONT_HERSHEY_SIMPLEX, (1000, 70), 3, 8, (0, 255, 0))
    draw_text(im0, "Drinking: " + str(watering_crickets), cv2.FONT_HERSHEY_SIMPLEX, (1000, 150), 3, 8, (0, 255, 255))
    # cv2.putText(im0, "Eating: " + str(eating_crickets), (1000, 70),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 7)
    # cv2.putText(im0, "Drinking: " + str(watering_crickets), (1000, 150),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 7)
    # cv2.putText(im0, "Outside: " + str(outside_cirkcets), (10, 320),
    #             cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 7)


def OCR(frame):

    cropframe = frame[0:60, 0:640]

    # Convert to GrayScale
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    level = 240
    cropframe[cropframe > level] = 250
    cropframe[cropframe <= level] = 1
    cropframe[cropframe == 250] = 0
    cropframe[cropframe == 1] = 255

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((3, 3), np.uint8)
    cropframe = cv2.dilate(cropframe, kernel, iterations=1)

    # just to show!
    cv2.imshow('crop', cropframe)

    result = pytesseract.image_to_string(cropframe)
    # print("result : {}".format(result))

    t = datetime.strptime(result, "%Y-%m-%d %H: %M: %S\n")
    # print("time:", t)

    return t


def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else:  # crosses midnight
        return check_time >= begin_time or check_time <= end_time

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        csv_folder='None',  # model.pt path(s) another model
        condition='dark',
        imgsz=(1920, 1080),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):


    print("============================================================")
    print("csv_folder : ", csv_folder)
    # todo: find out when it is none!
    if len(csv_folder) > 0:
        csv_path = csv_folder[0]
    print("csv_path : ", csv_path)

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Utilities
    frame_number = 0
    ocr_timestamp = None
    # default!, the real value will be determinded in the for loop
    if condition == 'dark':
        target_fps = 15
        target_ocr = 15 * 30
    elif condition == 'light':
        target_fps = 14
        target_ocr = 15 * 30
    else:
        target_fps = 15
        target_ocr = 15 * 30

    # initial CSV file
    header = ['timestamp', 'consuming_area', 'drinking_area', 'outside']
    data = []

    # just to fix!
    stage = 2

    # detecting area
    lower_y = 150
    upper_y = 900
    lower_x = 50
    upper_x = 1800

    # Object detection
    # todo: detect the feeder and water dispenser automatically
    # Basic statistics
    # dark condition!
    if condition == 'dark':
        feeder_x = 1255
        feeder_y = 449
        feeder_x2 = 1380
        feeder_y2 = 590
        feeder_r = 120
    elif condition == 'light':
        # light condition!
        feeder_x = 1295
        feeder_y = 469
        feeder_x2 = 1380
        feeder_y2 = 590
        feeder_r = 120
    else:
        # light dark condition!
        feeder_x = 844
        feeder_y = 290
        feeder_x2 = 1504
        feeder_y2 = 629
        feeder_r = 150

    feeder_w = feeder_x2 - feeder_x
    feeder_h = feeder_y2 - feeder_y

    if condition == 'dark':
        # dark condition!
        water_x = 625
        water_y = 321
        water_x2 = 917
        water_y2 = 521
    elif condition == 'light':
        # light condition
        water_x = 665
        water_y = 341
        water_x2 = 957
        water_y2 = 591
    else:
        # light dark condition
        water_x = 498
        water_y = 280
        water_x2 = 744
        water_y2 = 490

    water_w = water_x2 - water_x
    water_h = water_y2 - water_y

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:

        # skip the unused frame!
        if frame_number % target_fps != 0:
            frame_number += 1
            continue
        else :
            if ocr_timestamp is None or frame_number % target_ocr == 0:
                try:
                    ocr_timestamp = OCR(im0s)
                    print("ocr_timestamp(ocr)", ocr_timestamp)
                except Exception as e:
                    # ocr_timestamp = None
                    print("ocr_timestamp: error!", e)

            if ocr_timestamp is not None:
                ocr_timestamp = ocr_timestamp + timedelta(seconds=1)
                print("ocr_timestamp(timedelta)", ocr_timestamp)
            else:
                print("ocr: None")

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process detections
        outside_cirkcets = 0
        eating_crickets = 0
        watering_crickets = 0        
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image

                        # Mos: find the center of the box (of crickets)
                        pointX = math.floor((xyxy[0] + xyxy[2]) / 2.0)
                        pointY = math.floor((xyxy[1] + xyxy[3]) / 2.0)

                        # ignore upper
                        if pointY < 150:
                            continue

                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        center = (pointX, pointY)
                        pts.appendleft(center)

                        # Mos: draw a circle at the center of the box (of crickets)
                        cv2.circle(im0, center=(pointX, pointY),
                            radius=5, color=(0, 255, 0), thickness=-1)                                                

                        # note: because we have only 1 class, so every time detecting box == crickets
                        # feeding
                        if stage == 4 and (pointX > feeder_x and pointX < feeder_x + feeder_w) and (pointY > feeder_y and pointY < feeder_y + feeder_h):
                            eating_crickets += 1
                            # print("Eating detected")
                            # Mos: draw the box rectangle on the cricket
                            cv2.rectangle(im0, (math.floor(xyxy[0]), math.floor(xyxy[1])),
                                        (math.floor(xyxy[2]), math.floor(xyxy[3])), (0, 255, 255), 2)

                        elif stage == 2 and (pointX > feeder_x - feeder_r and pointX < feeder_x + feeder_r) and (pointY > feeder_y - feeder_r and pointY < feeder_y + feeder_r):

                            # todo: this has to be computed using the equation below!
                            # r^2 == (x-h)^2 + (y-k)^2
                            eating_crickets += 1
                            # print("Eating detected")
                            # Mos: draw the box rectangle on the cricket
                            cv2.rectangle(im0, (math.floor(xyxy[0]), math.floor(xyxy[1])),
                                            (math.floor(xyxy[2]), math.floor(xyxy[3])), (0, 255, 255), 2)

                        # water dispenser
                        elif (pointX > water_x and pointX < water_x + water_w) and (pointY > water_y and pointY < water_y + water_h):
                            watering_crickets += 1
                            # print("Watering detected")
                            # Mos: draw the box rectangle on the cricket
                            cv2.rectangle(im0, (math.floor(xyxy[0]), math.floor(xyxy[1])),
                                          (math.floor(xyxy[2]), math.floor(xyxy[3])), (255, 255, 0), 2)
                        else:
                            outside_cirkcets += 1
                            # print("Outside detected")
                            # Mos: draw the box rectangle on the cricket
                            cv2.rectangle(im0, (math.floor(xyxy[0]), math.floor(xyxy[1])),
                                      (math.floor(xyxy[2]), math.floor(xyxy[3])), (0, 255, 0), 2)

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # draw interface
            drawInterface(im0,
                            feeder_x,
                            feeder_y,
                            feeder_w,
                            feeder_h,
                            water_x,
                            water_y,
                            water_w,
                            water_h,
                            eating_crickets,
                            watering_crickets,
                            outside_cirkcets,
                            feeder_r)

            # input()
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # detect time of the frame!
        # if frame_number % target_fps == 0:
        #     # write data to csv file!
        #     csv_data = [str(ocr_timestamp), str(eating_crickets), str(watering_crickets), str(outside_cirkcets)]
        #     print("append : ", csv_data)
        #     data.append(csv_data)
        
        # increase frame number for the next round!
        frame_number += 1

        # # debug only
        # if (frame_number > 1200):
        #     break

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")


    # write to file!!
    csvname = "{}_timestamp.csv".format(os.path.basename(source))
    if csv_path is not None:
        csvname = os.path.join(csv_path, csvname)

    with open(csvname, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--csv-folder', nargs='+', type=str, default=None, help='path for csv folder')
    parser.add_argument('--condition', type=str, default='dark', help='(optional) select light condition (default: dark)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
