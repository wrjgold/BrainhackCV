import torch
import cv2
import csv
import pandas as pd
import gdown
from reID.model import SiameseNetwork
from reID.inference import infer
import os

yolo_url = "https://drive.google.com/file/d/1-YNxQ1EORYLcKVnKC1V5GZSB6aov0Ivn/view?usp=sharing"
yolo_output = "best.pt"
gdown.download( yolo_url, yolo_output, quiet=False)
yolo_model = model(yolo_output)

reid_model = SiameseNetwork()
reid_url = "https://drive.google.com/file/d/1MSUtLHjFWsUU6KIoI59C1fl-DmpSRU27/view?usp=sharing"
reid_output = "reid_model.pt"
gdown.download(reid_url, reid_output, quiet=False)
reid_model.load_state_dict(torch.load(reid_output))

submission_list = []
test_dir = "test"
suspect_dir = "suspects/content/drive/Shareddrives/ZINDI Data Science/ADPL/Competition Data/CV/Data Prep/Test (0-1599)/merged/crop"

def detect(image_name):
    """
    
    """
    image_path = os.path.join(test_dir, image_name)
    suspect_path = os.path.join(suspect_dir, image_name)
    objects_detected = []
    
    img = cv2.imread(image_path)
    suspect = cv2.imread(suspect_path)
    
    results = yolo_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxyn
            plushie = img[y1:y2, x1:x2]            
            classification, _ = infer(reid_model, plushie, suspect)
            objects_detected.append({'image_id': image_path, 'class': classification, 'ymin':y1, 'xmin':x1, 'ymax':y2, 'xmax':x2 })
    return objects_detected

with open(test_dir, 'r') as testSet:
    for image in testSet:
        submit = detect(image)
        for obj in submit:
            submission_list.append(obj)

submission_df = pd.DataFrame(submission_list)
submission_df.to_csv('submission.csv', index=False)



# with open('', 'w', newline='') as f:
#     writer = csv.writer(f)

            
