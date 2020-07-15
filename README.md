# MEEC_1920_LI2_G4
Face and voice recognition

Bibliotecas necessárias

tensorflow 1.7.0 (With tensorflow GPU)
Pillow 7.1.2
scikit-learn 0.20.0
PyQt5 5.14.2
matplotlib 3.2.1
scipy 1.1.0
PyAudio 0.2.11
librosa 0.6.3
audioread 2.1.8
SoundFile 0.9.0
opencv-python 4.2.0.34
Para correr o programa executar o main.py

Antes de treinar normalizar executando o ficheiro data_preprocess.py

Para re treinar executar o seguinte comando

python classifier.py TRAIN train_img model/20180402-114759.pb class/classifier_gpu.pkl --batch_size 10 --nrof_train_images_per_class 15 --use_split_dataset

Para classificar o dataset executar o comando

python classifier.py CLASSIFY train_img model/20180402-114759.pb class/classifier_gpu.pkl --batch_size 10 --nrof_train_images_per_class 15 --use_split_dataset
