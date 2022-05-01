import cv2
import sys
import os
import numpy as np
from set_color import *
import copy

def mk_video(video_name, split_path):
    # 파일경로 설정
    pred_file_path = split_path+'/'+video_name
    gt_file_path = './TCN/data/50salads/groundTruth/'+video_name+'.txt'
    save_path = './results_video/50salads/'
    # 파일 읽기
    with open(pred_file_path,'r') as f:
        lines = f.readlines()[1:]
        preds = lines[0].split()
        print('pred size : '+str(len(preds)))
    with open(gt_file_path,'r') as f:
        gts = f.readlines()
        gts = list(map(lambda x : x.strip(),gts))
        print('gt size : '+str(len(gts)))
    # 비디오 읽기
    cap = cv2.VideoCapture('./salads_video/'+video_name+'.avi')
    if not cap.isOpened():
        print('video is not opened.')
        sys.exit()

    # 프레임 수
    frame_cnt = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    effect_frames = int(fps*2)

    print(f'cnt : {frame_cnt}, fps: {fps}')
    delay = int(1000/fps)

    # 영상 가로세로 설정
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w : {w}, h : {h}')
    bar_h = 50
    h += (bar_h*2)

    # 비디오 코덱
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(save_path+video_name+'.avi',fourcc, fps, (w,h))

    # 예측 바 만들기
    pred_colors = []
    gt_colors = []
    for i in range(frame_cnt):
        pred_color = (0, 0, 0)
        gt_color = (0, 0, 0)
        if i < len(preds):
            pred_color = get_color(preds[i])
        if i < len(gts):
            gt_color = get_color(gts[i])
        pred_colors.append(pred_color)
        gt_colors.append(gt_color)
    print(f'preds : {len(pred_colors)}, gt : {len(gt_colors)}')
    pred_bar = np.array(pred_colors*bar_h).reshape((bar_h,frame_cnt,3)).astype('float32')
    gt_bar = np.array(gt_colors*bar_h).reshape((bar_h,frame_cnt,3)).astype('float32')
    # print(f'pred bar shape : {fixed_pred_bar.shape}')
    # print(f'gt bar shape : {fixed_gt_bar.shape}')
    for i in range(frame_cnt):
        ret, frame = cap.read()
        if not ret:
            break
        start = max(0,i-15)
        end = min(frame_cnt,i+15)
        now_pred_bar = copy.deepcopy(pred_bar)
        now_pred_bar[:,start:end] = [0,0,255]
        now_gt_bar = copy.deepcopy(gt_bar)
        now_gt_bar[:, start:end] = [0, 0, 255]
        fixed_pred_bar = cv2.resize(now_pred_bar, dsize=(w, bar_h), interpolation=cv2.INTER_AREA)
        fixed_gt_bar = cv2.resize(now_gt_bar, dsize=(w, bar_h), interpolation=cv2.INTER_AREA)
        new_frame = np.append(frame, fixed_pred_bar,axis=0)
        new_frame = np.append(new_frame, fixed_gt_bar, axis=0)
        new_frame = np.uint8(new_frame)
        out.write(new_frame)
    cap.release()
def main(split):
    split_path = './TCN/ms-tcn1/MS-TCN2results/50salads/split_'+split
    for name in os.listdir(split_path):
        mk_video(name,split_path)


if __name__ == "__main__":
    for i in range(1,6):
        main(str(i))
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--split', default='1')
    # args = parser.parse_args()
    # video_name = args.name
    # split = args.split