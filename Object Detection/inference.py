import torch
import cv2
import csv
import pandas as pd

yolo_model = torch.hub.load(wrjgold/BrainhackCV, 'best.pt', source='github')
reid_model = #
submission_list = []

def detect(image_path):
    """
    
    """
    objects_detected = []
    img = cv2.imread(image_path)

    results = yolo_model(img)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxyn
            plushie = img[y1:y2, x1:x2]
            #reid part
            objects_detected.append({'image_id': image_path, 'class': , 'ymin':y1, 'xmin':x1, 'ymax':y2, 'xmax':x2 })
    return objects_detected

with open('Test', 'r') as testSet:
    for image in testSet:
        submit = detect(image)
        for obj in submit:
            submission_list.append(obj)

submission_df = pd.DataFrame(submission_list)
submission_df.to_csv('submission.csv', index=False)



# with open('', 'w', newline='') as f:
#     writer = csv.writer(f)

            
