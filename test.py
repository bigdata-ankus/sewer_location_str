

import argparse
import time
from typing import List
import random
import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2
import torch
import csv
import torchvision.ops as ops
import os
from detection import Detector, Detection
import easyocr

from new_ocr.ocr_new import text_recognition_static
from new_ocr.ocr_new import text_recognition_dynamic
# coco class names
class_names = ['BC', 'BK', 'CC', 'CL', 'CM', 'DF', 'DG', 'DS', 'HL', 'JD', 'JF', 'JS', 'LD', 'LP', 'LS', 'PO', 'RI', 'SD', 'SG', 'TO']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(class_names))]


class ExportCSV:
    def __init__(self,csv_path):
        self.csv_path = csv_path


        if not os.path.isfile(self.csv_path):  # check whether file exists
            with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['file name', 'times', 'class', 'location x', 'location y', 'width', 'height', 'confidence','ocr'])

    def write_line(self, name, time, label, x, y, width, height, confidence,ocr):
        with open(self.csv_path, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([name, time, label, x, y, width, height, confidence,ocr])



def custom_NMS(detections):
    """
    params detections: [[x1,y1,x2,y2,confidence,classId], ...]
    """
    if detections:
        detections = torch.tensor(detections)
        detIds = ops.nms(detections[:, :4], detections[:, 4], 0.6)
        detections = detections[detIds]

        return detections.tolist()
    else:
        return detections


#
def draw_detection(img, detections: List[Detection], thickness=1):
    for det in detections:
        x_min, y_min, x_max, y_max = np.round(det.box).astype(np.int)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colors[int(det.classId)], thickness=thickness,lineType=cv2.LINE_AA)

        # t_size = cv2.getTextSize(class_names[det.classId], 0, fontScale=tl / 3, thickness=tf)[0]

        cv2.putText(img, class_names[det.classId] + ': ' + str(round(det.confidence, 2)), (x_min + 10, y_max - 10), 0, thickness/3, colors[int(det.classId)], thickness=2, lineType=cv2.LINE_AA)
    return img

def write_track_normal_frame(csv_write, video_time,frameId,ocr_result , ocr_static):


    csv_write.write_line(str(frameId), str(video_time), '', '', '',
                         '' , '' , '' ,
                         '{'+ocr_result[0]+'},{''},{'+ocr_static[0]+'},{'+ocr_static[1]+'},{'+ocr_static[2]+'},{'+ocr_static[3]+'},{'+ocr_static[4]+'}')


def write_track(csv_write, video_time,frameId,ocr_result , ocr_static, det):
    # x_min, y_min, x_max, y_max = track.get_box().flatten()
    x_min, y_min, x_max, y_max = np.round(det.box).astype(np.int)

    csv_write.write_line(str(frameId), str(video_time), class_names[int(det.classId)], str(x_min), str(y_min),
                         str(x_max - x_min), str(y_max - y_min), str(round(det.confidence, 2)),
                         '{'+ocr_result[0]+'},{''},{'+ocr_static[0]+'},{'+ocr_static[1]+'},{'+ocr_static[2]+'},{'+ocr_static[3]+'},{'+ocr_static[4]+'}')
    # fp.write(
    #     "{} {} {} {:.2f} {:.2f} {:.2f} {:.2f} {}\n".format(camId, frameId, int(track.trackId), x_min, y_min, x_max,
    #                                                        y_max, class_names[track.classId]))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="path to input video")
    parser.add_argument("-w", "--weights", default="detection/source/weights/best.pt")
    parser.add_argument("-in", "--image_num", default=3, help="set the number of images to extract per second")
    # parser.add_argument("--device", default="cuda")
    # parser.add_argument("--output_video", default="output/demo.mp4")
    # parser.add_argument("--output_track_result", default="output/track_results.txt")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    detector = Detector(args.weights, device=device)

    reader = easyocr.Reader(['ko', 'en'])
    reader_dynamic = easyocr.Reader(['en'])
    video_name = os.path.basename(args.input).split('.')[0]
    output_path = os.path.join('output', video_name)
    output_img_path = os.path.join(output_path, "images")
    # output_path = os.path.join("/mnt/CCTV/post_processing", video_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        # os.makedirs(os.path.join(output_path, 'images'))
        os.makedirs(output_img_path)
    path_out = os.path.join(output_path,'demo.mp4')
    #csv_write = ExportCSV(os.path.join(output_path, f'output_{args.image_num}.csv'))
    size = (1280, 720)

    test_video = args.input
    vs = cv2.VideoCapture(test_video)
    fps = vs.get(cv2.CAP_PROP_FPS)
    # writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    csv_write = ExportCSV(os.path.join(output_path,'output.csv'))

    count = 1
    count_frame = 1

    cap_static = cv2.VideoCapture(test_video)  # video_name is the video being called
    cap_static.set(1, cap_static.get(cv2.CAP_PROP_FRAME_COUNT)/2)  # Where frame_no is the frame you want
    ret, frame = cap_static.read()
    static_info = text_recognition_static(frame, reader)
    correct_distance_l = ['000']
    correct_distance_r = ['0']
    cap_static.release()

    # 초당 fps에 해당하는 모든 결함을 추출하지 않고 설정된 interval에 맞는 이미지만 추출하도록 수정
    # 초당 입력된 image_num 만큼만 출력하겠
    interval = fps // int(args.image_num)
    # os.makedirs(os.path.join(output_path, f'images_{args.image_num}'), exist_ok=True)

    while True:
        ret, frame = vs.read()
        video_time = (vs.get(cv2.CAP_PROP_POS_MSEC) / 1000)

        if not ret:
            break

        if count % interval == 0:
            start = time.time()
            detections = detector.detect(frame)
            detections = custom_NMS(detections)
            detections = [Detection(det) for det in detections]
            detect_timestamp = time.time()

            # onycom : get original frame image
            # frame = draw_detection(frame, detections,2)
            result_dynamic, correct_distance_l, correct_distance_r= text_recognition_dynamic(frame, reader_dynamic, correct_distance_l,correct_distance_r)
            cv2.imwrite(os.path.join(output_img_path, str(count) + '.jpg'), frame)
            if detections:
                cv2.imwrite(os.path.join(output_path,'images',str(count)+'.jpg'),frame)
                for det in detections:
                    write_track(csv_write, video_time, str(count), result_dynamic, static_info, det)
            else:
                write_track_normal_frame(csv_write, video_time, str(count), result_dynamic, static_info)
            # writer.write(frame)

        count += 1
        if count % 1000 == 0:
            print(f"count: {count}")

    vs.release()
    # writer.release()
    print('Finished!!!!')
