import cv2
import sys
import os
import numpy as np
from set_color import *
import copy
import argparse
from tqdm import tqdm

def mk_result_img(video_name, split_path, epoch_list):
    # set file path
    gt_file_path = './TCN/data/50salads/groundTruth/' + video_name + '.txt'
    save_path = './cmp_per_epoch/50salads/'

    # mk ground truth video & result
    with open(gt_file_path, 'r') as f:
        gts = f.readlines()
        gts = list(map(lambda x: x.strip(), gts))
        # print('gt size : ' + str(len(gts))) # 11115
    # read video
    cap = cv2.VideoCapture('./salads_video/' + video_name + '.avi')
    if not cap.isOpened():
        print('video is not opened.')
        sys.exit()
    # count frame
    frame_cnt = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'cnt : {frame_cnt}, fps: {fps}')

    # set width, height
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w : {w}, h : {h}')
    bar_h = 30
    line_h = 5
    h += bar_h * (len(epoch_list)+1)
    h += line_h * len(epoch_list)

    # video fourcc
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_path + video_name + '.avi', fourcc, fps, (w, h))

    # read pred files
    preds_list = []
    for epoch in epoch_list:
        pred_file_path = f'{split_path}/{epoch}ep/{video_name}'
        with open(pred_file_path, 'r') as f:
            lines = f.readlines()[1:]
            preds = lines[0].split()
            preds_list.append(preds)
            # print('pred size : ' + str(len(preds)))
    # 예측 바 만들기
    def mk_colorbar(data):
        bar = []
        for i in range(frame_cnt):
            color = (0, 0, 0)
            if i < len(data):
                color = get_color(data[i])
            bar.append(color)
        return np.array(bar * bar_h).reshape((bar_h, frame_cnt, 3)).astype('float32')

    gt_bar = mk_colorbar(gts)
    pred_bars = []

    for idx in range(len(epoch_list)):
        pred_bar = mk_colorbar(preds_list[idx])
        pred_bars.append(pred_bar)

    # print(f'pred bar shape : {fixed_pred_bar.shape}')
    # print(f'gt bar shape : {fixed_gt_bar.shape}')
    for i in tqdm(range(frame_cnt),desc="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        start = max(0,i-15)
        end = min(frame_cnt,i+15)
        new_frame = frame
        now_gt_bar = copy.deepcopy(gt_bar)
        now_gt_bar[:, start:end] = [0, 0, 255]
        fixed_gt_bar = cv2.resize(now_gt_bar, dsize=(w, bar_h), interpolation=cv2.INTER_AREA)
        new_frame = np.append(frame, fixed_gt_bar, axis=0)
        for idx in range(len(epoch_list)):
            black_line = np.zeros(shape=(line_h,w,3))
            new_frame = np.append(new_frame, black_line, axis=0)

            now_pred_bar = copy.deepcopy(pred_bars[idx])
            now_pred_bar[:,start:end] = [0,0,255]
            fixed_pred_bar = cv2.resize(now_pred_bar, dsize=(w, bar_h), interpolation=cv2.INTER_AREA)
            new_frame = np.append(new_frame, fixed_pred_bar, axis=0)

            
        
        new_frame = np.uint8(new_frame)
        out.write(new_frame)
    cap.release()
def main(result_path):
    epoch_list = list(map(lambda x: int(x.split('ep')[0]), os.listdir(result_path)))
    epoch_list.sort()
    print(epoch_list)
    # path setting
    temp_epoch = epoch_list[0]
    video_name_list = os.listdir(f'{result_path}/{temp_epoch}ep')
    for video_name in video_name_list:
        print(f'Start {video_name}')
        mk_result_img(video_name, result_path, epoch_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='./TCN/ms-tcn/MS-TCN2results/50salads')
    args = parser.parse_args()
    result_path  = args.result
    for split in range(1,6):
        split_path = result_path + '/split_' + str(split)
        main(split_path)