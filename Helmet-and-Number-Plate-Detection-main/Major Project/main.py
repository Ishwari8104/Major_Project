# import ultralytics
# ultralytics.checks()

# import torch
# print(torch.cuda.is_available())

from glob import glob
from itertools import chain
from collections import Counter
from pprint import pprint
import os
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import yaml
import pytesseract
import easyocr

id2class_map = {
    '0': 'with helmet',
    '1': 'without helmet',
    '2': 'rider',
    '3': 'number_plate'
}
main_path = os.getcwd()

def print_data_size(folder_type):
    data_size = len(glob(f'{main_path}/{folder_type}/labels/*.txt'))
    print(f'{folder_type} data count: {data_size}')

def print_class_count(folder_type):
    class_list = []
    for file in glob(f'{main_path}/{folder_type}/labels/*.txt'):
        class_list.append([row.split()[0] for row in open(file, "r")])
    counter = Counter(list(chain(*class_list)))
    print(f'-- data class count')
    pprint({f'{k}. {id2class_map[k]}':v for k, v in counter.items()})
    print()

def get_bbox_and_label(image_name, data_type='train', main_path=main_path):
    ''' get bbox and label information from label txt files '''

    # read file from path
    lbl_path = os.path.join(main_path, data_type, 'labels', f'{image_name}.txt')
    with open(lbl_path, 'r') as f:
        lines = f.readlines()

    # extract bboxes and labels from the label file
    bboxes = [
        [float(n) for n in line.split()[1:]]
        for line in lines
    ]
    labels = [id2class_map[line.split()[0]] for line in lines]

    return bboxes, labels

def load_image(image_name, data_type='train', main_path=main_path):
    img_path = os.path.join(main_path, data_type, 'images', f'{image_name}.jpg')
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_bbox_coordinates(img, bbox):
    # reference
    # https://medium.com/@miramnair/yolov7-calculating-the-bounding-box-coordinates-8bab54b97924

    img_height, img_width, _ = img.shape
    x_center, y_center, bbox_width, bbox_height = bbox

    # calculate the coordinates of the bounding box
    x_center_pixel = x_center * img_width
    y_center_pixel = y_center * img_height
    half_width = bbox_width * img_width / 2
    half_height = bbox_height * img_height / 2

    x_min = int(x_center_pixel - half_width)
    y_min = int(y_center_pixel - half_height)
    x_max = int(x_center_pixel + half_width)
    y_max = int(y_center_pixel + half_height)

    return x_min, y_min, x_max, y_max

class2color_map = {
    'with helmet': (0,255,128),
    'without helmet': (255,51,51),
    'rider': (51,255,255),
    'number_plate': (224,102,255)
}

def plot_image(image_name, data_type='train', class2color_map=class2color_map):
    img = load_image(image_name=image_name, data_type=data_type)
    bboxes, labels = get_bbox_and_label(image_name=image_name, data_type=data_type)
    for bbox, label in zip(bboxes, labels):

        # get bbox and label info
        color = class2color_map[label]
        x_min, y_min, x_max, y_max = get_bbox_coordinates(img, bbox)

        # add bounding box with rectangle
        img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)

        # add label info
        img = cv2.putText(
                img,
                label,
                (x_min, y_min + 10),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = color,
                thickness=2
            )
    plt.imshow(img)
    plt.show()

    # use results from `model.predict()` for plotting
def plot_pred_image(image_name, id2class_map=id2class_map, class2color_map=class2color_map):
    image_path = os.path.join(main_path, 'testimg', 'images', f'{image_name}.jpg')

    # get plot elements (bbox, labels) from `predict()` results
    results = model.predict(image_path)
    r = results[0]
    img = r.orig_img
    bboxes = r.boxes.xyxy.tolist()
    labels = [id2class_map[str(int(c))] for c in r.boxes.cls.tolist()]
    for bbox, label in zip(bboxes, labels):
        # get bbox and label info
        color = class2color_map[label]
        x_min, y_min, x_max, y_max = [int(n) for n in bbox]

        # add bounding box with rectangle
        img = cv2.rectangle(img,(x_min,y_min),(x_max,y_max), color, 2)

        # add label info
        img = cv2.putText(
                img,
                label,
                (x_min, y_min + 10),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.6,
                color = color,
                thickness=2
            )
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
reader = easyocr.Reader(['en']) 
# Function to perform OCR on a cropped number plate
def perform_ocr(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Use EasyOCR to detect text from the grayscale image
    result = reader.readtext(gray, detail=0)  # Set detail=0 to return just the text, not bounding boxes
    
    # Return the first result if available
    return result[0] if result else ''  # EasyOCR returns text as a list of strings

if __name__ == '__main__':
    # Specify the path to tesseract executable
    
    print_data_size('train')
    print_class_count('train')
    print_data_size('val')
    print_class_count('val')
    # plot_image(image_name='new101')

    # Load a COCO-pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Display model information (optional)
    model.info()

    # read the content of coco128.yaml
    # with open(os.path.join(main_path, 'coco128.yaml'), 'r') as file:
    #     print(file.read())

    # create a the yaml based on coco128 for model training
    data = {
        'train': os.path.join(main_path, 'train'),
        'val': os.path.join(main_path, 'val'),
        'nc': 4,
        'names': [
            'with helmet',
            'without helmet',
            'rider',
            'number plate'
        ]
    }

    with open(os.path.join(main_path,'data.yaml'), 'w') as file:
        yaml.dump(data, file)

    # check the content of data.yaml
    with open(os.path.join(main_path,'data.yaml'), 'r') as file:
        print(file.read())

    # disable wandb
    os.environ['WANDB_MODE'] = "disabled"


    # command to train the YOLOv8 model
    # model.train(
    #     data=os.path.join(main_path,'data.yaml'),
    #     epochs=50,
    #     workers=1,
    #     batch=8,
    # )

    # select the best model for checking prediction plot
    # the model is saved in best.pt directly after training
    model = YOLO(os.path.join(main_path,'runs/detect/train/weights/best.pt'))
    image_name = 'test'
    image_path = os.path.join(main_path, 'testimg', 'images', f'{image_name}.jpg')
    print('prediction ↓')
    plot_pred_image(image_name=image_name)
    # print('actual image ↓')
    # plot_image(image_name=image_name, data_type='val')

    # Read the input image
    img = cv2.imread(image_path)

    # Perform object detection
    results = model.predict(img)

    # Extract plot elements (bbox, labels) from `predict()` results
    # Extract plot elements (bbox, labels) from `predict()` results
r = results[0]
orig_img = r.orig_img  # Keep the original image
bboxes = r.boxes.xyxy.tolist()
labels = [id2class_map[str(int(c))] for c in r.boxes.cls.tolist()]

# Flags to check if 'without helmet' is detected
without_helmet_detected = False
number_plate_detected = False

# List to store the cropped number plates
number_plates = []
# List to store the OCR results
ocr_results = []

# First loop to check if 'without helmet' is detected
for bbox, label in zip(bboxes, labels):
    if label == 'without helmet':
        without_helmet_detected = True
        break  # Exit loop early since 'without helmet' is found

# If 'without helmet' is detected, proceed to apply OCR on number plates
if without_helmet_detected:
    print("No helmet detected, processing number plates for OCR...")

    # Second loop to find 'number_plate' labels and apply OCR
    for bbox, label in zip(bboxes, labels):
        if label == 'number_plate':
            x_min, y_min, x_max, y_max = [int(n) for n in bbox]
            # Crop the number plate
            cropped_plate = orig_img[y_min:y_max, x_min:x_max]

            # Perform OCR on the cropped number plate
            ocr_text = perform_ocr(cropped_plate)
            ocr_results.append(ocr_text)

            # Display the cropped number plate
            plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
            plt.title(f'OCR Result: {ocr_text}')
            plt.axis('off')
            plt.show()
else:
    print("Helmet detected, skipping number plate OCR.")

# Print all OCR results
if ocr_results:
    print("Detected Number Plates and OCR Results:")
    for i, ocr_text in enumerate(ocr_results, 1):
        print(f"Number Plate {i}: {ocr_text}")
else:
    print("No number plates processed for OCR.")
