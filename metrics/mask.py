import os
import cv2
import pandas as pd
import numpy as np

msk_path = "/"

def count_detected_points(pts, cnts, msk):
    detected_points = 0
    for c in cnts:
        tmp_msk = np.zeros(msk.shape, dtype=np.uint8)
        cv2.drawContours(tmp_msk, [c], -1, 1, thickness=cv2.FILLED)
        for pt in pts:
            if tmp_msk[int(pt[1]), int(pt[0])] == 1:
                detected_points += 1
                #break
    return detected_points

def pointToImage(pts, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius=1, color=1, thickness=-1)
    return img

def compute_scores_for_experiment(exp_dir, best_epoch=0):
    score_list = pd.read_csv(os.path.join(exp_dir, 'score_list.csv'))
  
    row = score_list.iloc[best_epoch]
    n = len(row['val_img_ids'])
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pts = 0
    img_list = row['val_img_ids']

    for i in range(n):
        msk = cv2.imread(os.path.join(msk_path, f"{img_list[i][0]}.png"))
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk = msk // 255
        cnts, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt = row['val_GT_PRED']['gt'][i]
        pts = row['val_GT_PRED']['pred'][i]
        
        pred_img = pointToImage(pts, msk.shape)
        total_gt += len(gt)
        total_pts += len(pts)
        
        if len(pts) > 0 and len(gt) > 0 and cnts:
            detected_pts = count_detected_points(pts, cnts, msk)
            total_tp += detected_pts
            total_fp += len(pts) - detected_pts
            detected_gt = count_detected_points(gt, cnts, msk)
            total_fn += len(gt) - detected_gt

    if total_pts > 0 and total_gt > 0:
        precision = total_tp / total_pts
        recall = total_tp / total_gt
        if precision + recall > 0:
            F1_score = 2 * (precision * recall) / (precision + recall)
        else:
            F1_score = 0
        CSI_score = total_tp / (total_tp + total_fp + total_fn)
    else:
        precision = 0
        recall = 0
        F1_score = 0
        CSI_score = 0

    return precision, recall, F1_score, CSI_score, best_epoch

experiments = [
    {"name": "LC Loss", "dir": "/", "best_epoch": 0},   
    {"name": "Fixed BB 7x7 0.0001", "dir": "/", "best_epoch": 0},
    {"name": "Fixed BB 7x7 0.05", "dir": "/", "best_epoch": 0},
    {"name": "LC FixedBB 10x10 0.0001", "dir": "/", "best_epoch": 0},
    {"name": "LC FixedBB 10x10 0.05", "dir": "/", "best_epoch": 0},
]

for experiment in experiments:
    precision, recall, F1_score, CSI_score = compute_scores_for_experiment(experiment['dir'], best_epoch=experiment['best_epoch'])
    print(f"Experiment: {experiment['name']}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {F1_score}, CSI: {CSI_score}")
    print("-" * 100)
