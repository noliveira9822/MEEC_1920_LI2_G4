from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import sys
import time
import pickle
import facenet
import detect_face
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import numpy as np
import tensorflow as tf
from scipy import misc

modeldir = 'Facenet-Real-time-face-recon/model/20170511-185253.pb'
classifier_filename = 'Facenet-Real-time-face-recon/class/classifier.pkl'
npy='Facenet-Real-time-face-recon/npy'
train_img="Facenet-Real-time-face-recon/train_img"
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
    qtimerCamera.start(50)
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
    #face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    curTime = time.time() + 1  # calc fps
    timeF = frame_interval

    if (c % timeF == 0):
        find_results = []
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        print('Face Detected: %d' % nrof_faces)

        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(frame.shape)[0:2]

            cropped = []
            scaled = []
            scaled_reshape = []
            bb = np.zeros((nrof_faces, 4), dtype=np.int32)

            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))

                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                # inner exception
                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                    window.status.setText('face is too close')
                    continue

                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                cropped[i] = facenet.flip(cropped[i], False)
                scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                       interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                print(predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                # print(best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                print(best_class_probabilities)

                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (105, 189, 45), 1)
                window.image_input.setPixmap(img2map(frame))
                # boxing face

                # plot result idx under box
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                print('Result Indices: ', best_class_indices[0])
                #print(HumanNames)
                for H_i in HumanNames:
                    #print(H_i)
                    if HumanNames[best_class_indices[0]] == H_i:
                        result_names = HumanNames[best_class_indices[0]]
                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 255), thickness=1, lineType=2)
                        print(result_names)
                        window.status.setText(result_names)
        else:
            window.status.setText('Unable to align')

#    if face_cascade.empty():
#        window.status.setText("Cannot detect faces")
#        window.image_input.setPixmap(img2map(image))
#    else:
#        faces = face_cascade.detectMultiScale(image, 1.3, 6)
#        for (x, y, w, h) in faces:
#            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#            window.image_input.setPixmap(img2map(image))

if __name__ == "__main__":
    cap = cv2.VideoCapture()
    app = QtWidgets.QApplication(sys.argv)
    window = uic.loadUi("mainWindow.ui")
    window.image_input.setPixmap(QPixmap("black.png"))

    window.image_crop.setPixmap(QPixmap("black.png"))
    window.button_cameraOn.clicked.connect(cameraOn)
    window.button_cameraOff.clicked.connect(cameraOff)
    qtimerCamera = QTimer()
    qtimerCamera.timeout.connect(grabImageInput)

    window.show()
    app.exec()
