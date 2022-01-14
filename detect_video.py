import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from urllib.request import urlopen


fframework = 'tf'                                 # is model tf or tflite
fweights = './checkpoints/yolov4-tiny-416'        # path to weights file
fsize = 416                                       # size of video frame
ftiny = True                                      # is model yolo tiny
fmodel = 'yolov4'                                 # is model yolov4 or v3
fvideo = './data/video/cars-on-highway.mp4'       # input video path
foutput = './detections/results.avi'              # output video path
foutput_format = 'XVID'                           # video codec
fiou = 0.45                                       # iou threshold
fscore = 0.25                                     # score threshold
fdont_show = False                                # true if not to show video frames


def detect_ip_webcam(url, infer):
    while True:
        img_s= urlopen(url)
        img_mat= np.array(bytearray(img_s.read()),dtype=np.uint8)
        img= cv2.imdecode(img_mat,-1)

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (fsize, fsize))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if fframework == 'tflite':
            pass
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=fiou,
            score_threshold=fscore
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not fdont_show:
            cv2.imshow("result", result)
        
        # if foutput:
        #     out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

def detect_web_video():
    pass

def main(opt_dict):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(ftiny, fmodel)
    input_size = fsize
    video_path = fvideo

    if fframework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=fweights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(fweights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']


    cap_opt = list(opt_dict.keys())[0]

    if cap_opt == 1:
        url = opt_dict[cap_opt]
        detect_ip_webcam(url, infer)
        exit()
        
    elif cap_opt == 2:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)

    out = None

    if foutput:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*foutput_format)
        out = cv2.VideoWriter(foutput, codec, fps, (width, height))

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if fframework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if fmodel == 'yolov3' and ftiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=fiou,
            score_threshold=fscore
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if not fdont_show:
            cv2.imshow("result", result)
        
        if foutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':

    opt_dict = {}

    print('\nSelect:\n1. IP webcam\t2. Webcam\t3. Video')
    cap_option = int(input())

    if cap_option == 1:
        print('Enter IP address: ')
        url = input()
        url += '/shot.jpg'
        opt_dict[cap_option] = url
    else:
        opt_dict[cap_option] = -1
    
    main(opt_dict)
  
