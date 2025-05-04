import os
import pyodbc
import cv2, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog
# from tensorflow.keras.applications import VGG16
# from tensorflow  import keras
# import tensorflow as tf

from keras._tf_keras.keras.applications.vgg16 import VGG16, preprocess_input
from keras._tf_keras.keras.preprocessing.image import img_to_array

model = VGG16(weights='imagenet', include_top=True,input_shape=(224,224,3))
# def extract_features(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (224, 224))
#     rgb_mean = np.mean(img, axis=(0, 1))
#     hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     hog = hog_feature(img)  # Hàm giả định
#     cnn_features = model.predict(preprocess_image(img))  # Hàm giả định
#     return np.concatenate([rgb_mean, hist.flatten(), hog, cnn_features.flatten()])
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    rgb_mean = np.mean(img, axis=(0, 1))
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_desc = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    cnn_features = model.predict(preprocess_image(img))  # Hàm giả định
    return np.concatenate([rgb_mean, hist.flatten(), hog_desc, cnn_features.flatten()])
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def get_sql_server_connection():
    try:
        connection=pyodbc.connect('DRIVER={SQL Server};'+
                                'Server=DESKTOP-S86ALTA;'+
                                'Database=bird_db;'+
                                'Trusted_Connection=True')
        print('Connected to database')
    except pyodbc.Error as ex:
        print('Connection failed', ex)
    return connection
def create_table():
    conn = get_sql_server_connection()
    cursor = conn.cursor()
    cursor.execute('''
        IF NOT EXISTS (
            SELECT * FROM sysobjects WHERE name='bird_images' AND xtype='U'
        )
        CREATE TABLE bird_images (
            id INT PRIMARY KEY,
            image_path NVARCHAR(255),
            features VARBINARY(MAX)
        )
    ''')
    conn.commit()
    conn.close()
def build_database():
    conn = get_sql_server_connection()
    cursor = conn.cursor()
    create_table()
    

    root_folder = 'img/birds_resized256'

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(dirpath, filename)
                
                path = file_path
                try:
                    features = extract_features(path).astype(np.float32)
                    cursor.execute(
                        "INSERT INTO bird_images (image_path, features) VALUES (?, ?)",
                        path, features.tobytes()
                    )
                    print(f"Đã thêm ảnh: {path}")
                except Exception as e:
                    print(f"Lỗi xử lý ảnh {path}: {e}")

    conn.commit()
    conn.close()
def search_image(input_image_path):
    input_features = extract_features(input_image_path).astype(np.float32)
    conn = get_sql_server_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT image_path, features FROM bird_images")

    results = []
    for row in cursor.fetchall():
        db_features = np.frombuffer(row[1], dtype=np.float32)
        similarity = cosine_similarity([input_features], [db_features])[0][0]
        results.append((row[0], similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]  # Trả về 3 ảnh giống nhất
def show_img(image_path):
    image_path=cv2.imread(image_path)
    cv2.imshow('Ảnh của tôi',image_path)
    cv2.waitKey(0)

input_img = 'img/birds_resized256\Clark_Nutcracker\Clark_Nutcracker_04.jpg'
# input_img = cv2.imread(input_img)
# input_img = cv2.resize(input_img, (224,224))
# top_3 = search_image(input_img)
# for path, sim in top_3:
#     print(f'Image: {path}, Similarity: {sim}')

show_img('img/birds_resized256\Red_winged_Blackbird\Red_Winged_Blackbird_0007_3706.jpg')

# build_database()