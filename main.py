import cv2
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import sqlalchemy
from easyocr import Reader
import datetime
import re
import time
import logging
import os

# Setup Database credentials here
postgres_host = ""
con = {
    "username":"",
    "password":"",
    "connectstr":f"jdbc:postgresql://{postgres_host}:5433/",
    "database":"",
    "host":f"{postgres_host}",
    "type":"public"
}

# Specify yolo model name here
# yolo_model = 'yolov3-tiny'
yolo_model = 'yolov3'

# check if postgres_host, con['userneame'] and con['password'] are set
if not postgres_host or not con['username'] or not con['password'] or not con['database']:
    raise Exception("Please set postgres_host, con['username'] and con['password'] and con['database']")

image_folder = (os.environ['DISK']+'/yolo') if 'DISK' in os.environ else 'images.nosync'
model_folder = (os.environ['MODEL']+'/yolo') if 'MODEL' in os.environ else '.'


classes = ["person"]
# additional_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
# ]

# what to detect
detection_classes = ['person']

def get_output_layers(net):
    layers = net.getLayerNames()
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    if len(classes) <= class_id:
        return
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def query(query, db_conn):
    conn = psycopg2.connect(
        host=db_conn["host"],
        database=db_conn["database"],
        user=db_conn["username"],
        password=db_conn["password"],
        port=5433
    )
    # cur = conn.cursor()
    data = pd.read_sql_query(query, conn)
    conn.commit()
    # cur.close()
    return data

def insert_or_update(db_conn, df, table, conflict_rows=[], dtypes={}):
    try:
        conn = psycopg2.connect(
            host=db_conn["host"],
            database=db_conn["database"],
            user=db_conn["username"],
            password=db_conn["password"],
            port=5433
        )
        cur = conn.cursor()
        for row in df:
            insert_or_update_wrapped(cur, table, row, conflict_rows, dtypes)
        conn.commit()
        cur.close()
        if conn: conn.close()
    except (Exception, psycopg2.DatabaseError):
        logging.exception("Error executing PostgreSQL")


def insert_or_update_wrapped(cur, table, data, conflict_rows=[], dtypes={}):
    data = {k:v.replace("'","''") if type(v) is str else v for (k,v) in data.items()}
    keys = ", ".join(data.keys())
    is_number = lambda x:(type(x) is int or type(x) is float)
    get_dtype = lambda x:(f"::{dtypes[x]}") if x in dtypes else ""
    values = ", ".join(list(map(lambda x: f"{x[1]}" if is_number(x) else ("null" if x[1] is None else f"'{x[1]}'")+get_dtype(x[0]), data.items())))
    conflict_rows_s = ", ".join(conflict_rows)
    non_conflict_values = ", ".join(list(map(lambda x: f"{x[0]}={x[1]}" if is_number(x[1]) else (f"{x[0]}=null" if x[1] is None else  f"{x[0]}='{x[1]}'"), filter(lambda tpl: not tpl[0] in conflict_rows, data.items()))))
    if len(conflict_rows) == 0:
        query = f"""
insert into {table} ({keys}) 
values({values});"""
    else:
        query = f"""
insert into {table} ({keys}) 
values({values})
on conflict ({conflict_rows_s}) do
update set {non_conflict_values}"""
    cur.execute(query)


df_cams = query("select * from webcam_urls where debug = false", con)


COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
default_date = datetime.datetime.now()

detection_results = []

for cam in df_cams.iterrows():
    try:
        start = time.time()

        id = cam[1]["id"]

        # prepare image data and model
        resp = urllib.request.urlopen(cam[1]["url"])
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        net = cv2.dnn.readNet(model_folder+f'/{yolo_model}.weights', model_folder+f'/{yolo_model}.cfg')

        # try to parse date and time with ocr
        reader = Reader(['de', 'en'])
        text_results = reader.readtext(image)
        s = " ".join(map(lambda x: x[1], text_results)).replace(",", "")
        re_date = re.search(r'\d{2}.\d{2}.\d{2,4}', s)
        if re_date:
            s = s.replace(re_date.group(0), "")
        re_time = re.search(r'\d{2}(:|\.)\d{2}', s)
        parsed_date = None
        if re_date and re_time:
            date = re_date.group(0).replace(" ", ".") + " " + re_time.group(0).replace(" ", ".")
            try:
                parsed_date = datetime.datetime.strptime(date, '%d.%m.%Y %H.%M')
            except ValueError:
                try:
                    parsed_date = datetime.datetime.strptime(date, '%d.%m.%Y %H:%M')
                except ValueError as e:
                    print("Failed to parse date", date)
                    print(e)


        # segment image into smaller images
        count_x = image.shape[1]//416
        count_y = image.shape[0]//416
        scale = 0.00392*6
        Width = w = image.shape[1] // count_x
        Height = h = image.shape[0] // count_y
        images = []
        for x in range(count_x):
            for y in range(count_y):
                i = image[y*h:min(y*h+h, image.shape[0]), x*w:min(x*w+w, image.shape[1])]
                images.append(i)

        # prepare nn input
        blobs = [cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False) for image in images]

        detections = 0
        results = []
        for ix, blob in enumerate(blobs):
            net.setInput(blob)
            outs = net.forward(get_output_layers(net))

            # initialization
            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = 0.5
            nms_threshold = 0.4

            # search for detections with confidence > 0.5
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            detections += len(boxes)
            if len(boxes) > 0:
                results.append((ix, boxes, class_ids, confidences))

        # get indices of classes to detect and prepare data for db
        class_indices = list(map(lambda x: classes.index(x), detection_classes))
        detection_results += (new_results := [{
            "webcam":id,
            "date": parsed_date or default_date,
            "detections": len(list(filter(lambda x: x[2][0] in class_indices, results)))
        }])

        # store raw images locally
        folder = image_folder + f'/{id}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        if len(new_results) > 0:
            path = image_folder + f'/{id}/raw_{parsed_date or default_date}.jpg'
            cv2.imwrite(path, image)
        
        # draw bounding boxes
        for result in results:
            ix, boxes, class_ids, confidences = result
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            for i in indices:
                i = i
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_bounding_box(images[ix], 
                                class_ids[i], 
                                confidences[i], 
                                round(x),     round(y), 
                                round(x+w),   round(y+h))
        
        # store raw images with detections locally
        if len(new_results) > 0:
            path = image_folder + f'/{id}/{parsed_date or default_date}.jpg'
            cv2.imwrite(path, image)
            
        # print results
        ls = list(map(lambda x: classes[x[2][0]] if x[2][0] in class_indices else "-", results))
        print(cam[1]["url"], len(new_results), ls, str(time.time()-start)+"s", parsed_date or default_date)
    except Exception as e:
        print(e)

# localize timestamps and store in db
df = pd.DataFrame(detection_results)
df.index = df.date
df.index = df.index.tz_localize('Europe/Berlin')
df.index = df.index.tz_convert('UTC')
df.date = df.index
insert_or_update(con, df.to_dict(orient='records'), 'webcam_detections', ['webcam', 'date'])
print(df)
