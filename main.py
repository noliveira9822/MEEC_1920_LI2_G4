import os
import cv2
import sys
import pickle
import facenet
import detect_face
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import numpy as np
import tensorflow as tf
from scipy import misc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
train_img="./train_img"
c = 0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()

        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

def img2map(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap

def cameraOn():
    window.status.setText("Turning camera On")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    qtimerCamera.start(30)
    window.status.setText("Capturing image...")

def cameraOff():
    window.status.setText("Turning camera Off")
    if cap.isOpened():
        qtimerCamera.stop()
        cap.release()
    window.image_input.setPixmap(QPixmap("black.png"))
    window.status.setText("Camera Off")

def grabImageInput():
    if not cap.isOpened():
        cap.open(0)
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
    timeF = frame_interval

    if (c % timeF == 0):
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)  # MTCNN
        nfaces = bounding_boxes.shape[0]

        if nfaces > 0:
            det = bounding_boxes[:, 0:4]
            cropped = []
            scaled = []
            scaled_reshape = []
            bbox = np.zeros((nfaces, 4), dtype=np.int32)
            for i in range(nfaces):
                emb_array = np.zeros((1, embedding_size))
                bbox[i][0] = det[i][0]
                bbox[i][1] = det[i][1]
                bbox[i][2] = det[i][2]
                bbox[i][3] = det[i][3]

                if bbox[i][0] <= 0 or bbox[i][1] <= 0 or bbox[i][2] >= len(frame[0]) or bbox[i][3] >= len(frame):
                    window.status.setText('face is too close')
                    window.image_input.setPixmap(img2map(frame))
                    continue

                cropped.append(frame[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2], :])
                # cropped[i] = facenet.flip(cropped[i], False)
                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])  # Pixel range normalization
                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))

                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)

                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probability = predictions[np.arange(len(best_class_indices)), best_class_indices]

                cv2.rectangle(frame, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), (105, 189, 45), 1)
                for H_i in HumanNames:
                    if HumanNames[best_class_indices[0]] == H_i:
                        if best_class_probability > 0.5:
                            result_name = HumanNames[best_class_indices[0]]
                            cv2.putText(frame, result_name, (bbox[i][0], bbox[i][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (105, 195, 45), 1)
                        else:
                            window.certeza.setText("Unnable to give a certain guess")

                window.image_input.setPixmap(img2map(frame))
        else:
            window.status.setText('Unable to align')
            window.image_input.setPixmap(img2map(frame))

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    app = QtWidgets.QApplication(sys.argv)
    window = uic.loadUi("mainWindow.ui")
    window.image_input.setPixmap(QPixmap("black.png"))
    window.button_cameraOn.clicked.connect(cameraOn)
    window.button_cameraOff.clicked.connect(cameraOff)
    qtimerCamera = QTimer()
    qtimerCamera.timeout.connect(grabImageInput)

    window.show()
    app.exec()
