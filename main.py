import os
import cv2.cv2 as cv2
import sys
import pickle
import facenet
import detect_face
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from matplotlib.backends.qt_compat import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import misc
import sound_utils
import pyaudio
import wave
from librosa.util import normalize
import threading
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only to remove the AVX2 instruction warning

model_dir = './model/20180402-114759.pb'
classifier_filename = './class/classifier_gpu.pkl'
npy = './npy'
train_img = "./train_img"

commands_model = './sound_models/mlp_classifier_commands_0407.model'
groups_model = './sound_models/mlp_classifier_groups_0407.model'

command_model = pickle.load(open(commands_model, "rb"))
group_model = pickle.load(open(groups_model, 'rb'))

cam_number = 0
c = 0
chunk = 1024
fs = 44100
sample_format = pyaudio.paInt16

temp_array = np.zeros((1, chunk))
p = pyaudio.PyAudio()
Stream = p.open(format=sample_format, channels=1, rate=fs, frames_per_buffer=chunk, input=True)
p.close(Stream)

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    # settings = tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=3,
    #                          log_device_placement=False)
    # sess = tf.Session(config=settings)
    # Easter egg
    print('████░ █████░ ████░ ████░ ██░   █░ ████░ █████░')
    print('█░    █░  █░ █░    █░    █░█░  █░ █░      █░')
    print('███░  █████░ █░    ███░  █░ █░ █░ ███░    █░')
    print('█░    █░  █░ █░    █░    █░  █░█░ █░      █░')
    print('█░    █░  █░ ████░ ████░ █░   ██░ ████░   █░')
    print('           WEBCAM FACE IDENTIFIER')
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 160  # 182
        input_image_size = 160

        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading image models...')
        facenet.load_model(model_dir)
        print('Loading sound models...')
        print('Model filename: ' + commands_model)
        print('Model filename: ' + groups_model)
        print('Loading classes...')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)


def img2map(image):
    height, width, channel = image.shape
    bytes_per_line = 3 * width
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


def cameraOn():
    window.status.setText("Turning camera On")
    if not cap.isOpened():
        cap.open(cam_number)
    qtimerCamera.start(80)
    cap.set(3, 320)
    cap.set(4, 240)
    window.status.setText("Capturing image...")


def cameraOff():
    window.status.setText("Turning camera Off")
    if cap.isOpened():
        qtimerCamera.stop()
        cap.release()

    window.image_input.setPixmap(QPixmap("black.png"))
    window.status.setText("Camera Off")
    window.certeza.setText(" ")


def grabImageInput():
    if not cap.isOpened():
        cameraOn()
    ret, frame = cap.read()  # Returns the frame from camera
    time_f = frame_interval

    if (c % time_f == 0):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                # scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                # scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                scaled.append(
                    cv2.resize(cropped[i], (input_image_size, input_image_size), interpolation=cv2.INTER_NEAREST))
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
                        if best_class_probability > 0.50:
                            result_name = HumanNames[best_class_indices[0]]
                            cv2.putText(frame, result_name, (bbox[i][0], bbox[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (105, 195, 45), 1)
                            window.certeza.setText("Probability: %.2f %%" % (best_class_probability[0] * 100))
                        else:
                            window.certeza.setText("Unknown Face")

                window.image_input.setPixmap(img2map(frame))
        else:
            window.status.setText('Unable to align')
            window.certeza.setText(' ')
            window.image_input.setPixmap(img2map(frame))


def sound_read(in_data, frame_count, time_info, status):
    dados = np.frombuffer(in_data, dtype=np.int16)
    data = dados.astype("float")
    print(data.shape)
    window.plotgraph(data)

    features = sound_utils.feature_extract_classify(data, fs, mfcc=True, chroma=True, mel=True)
    predictions_command = command_model.predict(features.reshape(1, -1))
    predictions_group = group_model.predict(features.reshape(1, -1))

    print(predictions_command)

    window.sound_word_guess.setText("Comando: " + str(predictions_command[0]))
    window.sound_group_guess.setText("Grupo: " + str(predictions_group[0]))

    return data, pyaudio.paContinue


def init_recording():
    window.sound_status.setText("Recording...")
    # seconds = 1.5
    global Stream
    Stream = p.open(format=sample_format, channels=1, rate=fs, frames_per_buffer=fs * 2, input=True,
                    stream_callback=sound_read)
    #window.plotgraph(Stream.read(1024))

def stop_recording():
    window.sound_status.setText("Stopping recording...")
    global Stream
    p.close(Stream)
    window.sound_status.setText("Recording stopped.")
    window.sound_word_guess.setText(" ")
    window.sound_group_guess.setText(" ")

def quit_application():
    qapp.exit(0)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        uic.loadUi("mainWindow.ui", self)
        self.image_input.setPixmap(QPixmap("black.png"))
        self.button_cameraOn.clicked.connect(cameraOn)
        self.button_cameraOff.clicked.connect(cameraOff)

        self.start_recording.clicked.connect(init_recording)
        self.stop_recording.clicked.connect(stop_recording)

        self.quit_btn.clicked.connect(quit_application)
        self.image_input.setScaledContents(True)
        self.show()

    def plotgraph(self, stream):
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(stream)
        self.MplWidget.canvas.draw()

if __name__ == "__main__":
    print('Loading user interface...')
    cap = cv2.VideoCapture(cam_number)
    qapp = QtWidgets.QApplication(sys.argv)
    window = ApplicationWindow()
    qtimerCamera = QTimer()
    qtimerCamera.timeout.connect(grabImageInput)
    print('Done!')
    qapp.exec()
