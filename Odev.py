import sys
from PyQt5.QtWidgets import QApplication, QGridLayout, QMessageBox, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog, QStackedWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
import cv2
import math
import numpy as np
import pandas as pd
from skimage.measure import regionprops, label, shannon_entropy

class Odev1Page(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        bold_font = QFont("Arial", 14, QFont.Bold)

        self.label_odev = QLabel("Ödev-1", self)
        self.label_odev.setFont(bold_font)
        self.label_odev.setStyleSheet("color: #333; padding: 5px;")
        left_layout.addWidget(self.label_odev)
        
        self.label_id = QLabel("221229032", self)
        self.label_id.setFont(bold_font)
        self.label_id.setStyleSheet("color: #777; padding: 5px;")
        left_layout.addWidget(self.label_id)

        self.label_name = QLabel("Zehra Küçüker", self)
        self.label_name.setFont(bold_font)
        self.label_name.setStyleSheet("color: #555; padding: 5px;")
        left_layout.addWidget(self.label_name)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 500px; background-color: #f5f5f5;")
        self.image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        left_layout.addWidget(self.image_label)

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 300px; background-color: #f5f5f5;")
        self.processed_image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        right_layout.addWidget(self.processed_image_label)

        self.button = QPushButton("Görsel Yükle", self)
        self.button.setFont(QFont("Arial", 10))
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.button.clicked.connect(self.load_image)
        left_layout.addWidget(self.button)

        self.grayscale_button = QPushButton("Gri Tonlama", self)
        self.grayscale_button.clicked.connect(self.grayscale)
        right_layout.addWidget(self.grayscale_button)

        self.blur_button = QPushButton("Bulanıklık", self)
        self.blur_button.clicked.connect(self.blur)
        right_layout.addWidget(self.blur_button)

        self.color_change_button = QPushButton("Renk Uzayı Değiştirme", self)
        self.color_change_button.clicked.connect(self.color_change)
        right_layout.addWidget(self.color_change_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        self.setGeometry(200, 200, 1000, 800)
        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Bir Görsel Seçin", "", "Tüm Dosyalar (*);;JPEG (*.jpg *.jpeg);;PNG (*.png)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def grayscale(self):
        if self.image_path:
            img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            self.processed_image(img)

    def blur(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.GaussianBlur(img, (15, 15), 0)
            self.processed_image(img)

    def color_change(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self.processed_image(img)

    def processed_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        h, w = img.shape[:2]
        q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.processed_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

class Odev2Page(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.zoom_factor = 1.0
        self.rotation_angle = 0
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        bold_font = QFont("Arial", 14, QFont.Bold)

        self.label_odev = QLabel("Ödev - 2", self)
        self.label_odev.setFont(bold_font)
        self.label_odev.setStyleSheet("color: #333; padding: 5px;")
        left_layout.addWidget(self.label_odev)

        self.label_id = QLabel("221229032", self)
        self.label_id.setFont(bold_font)
        self.label_id.setStyleSheet("color: #777; padding: 5px;")
        left_layout.addWidget(self.label_id)

        self.label_name = QLabel("Zehra Küçüker", self)
        self.label_name.setFont(bold_font)
        self.label_name.setStyleSheet("color: #555; padding: 5px;")
        left_layout.addWidget(self.label_name)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 500px; background-color: #f5f5f5;")
        self.image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        left_layout.addWidget(self.image_label)

        self.button = QPushButton("Görsel Yükle", self)
        self.button.setFont(QFont("Arial", 10))
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.button.clicked.connect(self.load_image)
        left_layout.addWidget(self.button)

        main_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 300px; background-color: #f5f5f5;")
        self.processed_image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        right_layout.addWidget(self.processed_image_label)

        buttons = [
            ("Büyüt (Nearest Neighbor)", self.enlarge_image),
            ("Küçült (Average)", self.shrink_image),
            ("Zoom In (Bilinear)", self.zoom_in),
            ("Zoom Out (Bilinear)", self.zoom_out),
            ("Döndür (Bilinear)", self.rotate),
        ]

        for button_text, button_function in buttons:
            button = QPushButton(button_text, self)
            button.clicked.connect(button_function)
            right_layout.addWidget(button)

        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Resim Dosyaları (*.jpg *.png *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image, self.image_label)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def enlarge_image(self):
        if self.image is None: return
        enlarged_img = cv2.resize(self.image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
        self.display_image(enlarged_img, self.processed_image_label)

    def shrink_image(self):
        if self.image is None: return
        shrunk_img = cv2.resize(self.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        self.display_image(shrunk_img, self.processed_image_label)

    def zoom_in(self):
        if self.image is None: return
        self.zoom_factor *= 1.2
        if self.zoom_factor > 5.0:
            self.zoom_factor = 5.0
        self.update_image()

    def zoom_out(self):
        if self.image is None: return
        self.zoom_factor /= 1.2
        if self.zoom_factor < 0.2:
            self.zoom_factor = 0.2
        self.update_image()

    def update_image(self):
        if self.image is None: return
        height, width = self.image.shape[:2]
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)

        resized_img = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.display_image(resized_img, self.processed_image_label)

    def rotate(self):
        if self.image is None: return
        height, width = self.image.shape[:2]

        self.rotation_angle += 90
        if self.rotation_angle >= 360:
            self.rotation_angle = 0

        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1)
        rotated_img = cv2.warpAffine(self.image, rotation_matrix, (width, height))

        self.display_image(rotated_img, self.processed_image_label)
        
class Odev3Page(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        bold_font = QFont("Arial", 14, QFont.Bold)

        self.label_odev = QLabel("Ödev - 3 : S - Curve Metodu", self)
        self.label_odev.setFont(bold_font)
        self.label_odev.setStyleSheet("color: #333; padding: 5px;")
        left_layout.addWidget(self.label_odev)

        self.label_id = QLabel("221229032", self)
        self.label_id.setFont(bold_font)
        self.label_id.setStyleSheet("color: #777; padding: 5px;")
        left_layout.addWidget(self.label_id)

        self.label_name = QLabel("Zehra Küçüker", self)
        self.label_name.setFont(bold_font)
        self.label_name.setStyleSheet("color: #555; padding: 5px;")
        left_layout.addWidget(self.label_name)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 500px; background-color: #f5f5f5;")
        self.image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        left_layout.addWidget(self.image_label)

        self.load_button = QPushButton("Görsel Yükle", self)
        self.load_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 300px; background-color: #f5f5f5;")
        self.processed_image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        right_layout.addWidget(self.processed_image_label)

        buttons = [
            ("Standart Sigmoid", self.standard_sigmoid),
            ("Yatay Kaydırılmış Sigmoid", self.shifted_sigmoid),
            ("Eğimli Sigmoid", self.slope_sigmoid),
            ("Özel Fonksiyon", self.custom_function)
        ]

        for text, func in buttons:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            right_layout.addWidget(btn)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Resim Dosyaları (*.jpg *.png *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image, self.image_label)

    def display_image(self, img, label):
        q_img = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def apply_function(self, func):
        if self.image is not None:
            normalized = self.image / 255.0
            result = func(normalized)
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            self.display_image(result, self.processed_image_label)

    def standard_sigmoid(self):
        self.apply_function(lambda x: 1 / (1 + np.exp(-x)))

    def shifted_sigmoid(self):
        self.apply_function(lambda x: 1 / (1 + np.exp(-(x - 0.5))))

    def slope_sigmoid(self):
        self.apply_function(lambda x: 1 / (1 + np.exp(-10 * (x - 0.5))))

    def custom_function(self):
        self.apply_function(lambda x: np.power(x, 2))

class Odev4Page(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        bold_font = QFont("Arial", 14, QFont.Bold)

        self.label_odev = QLabel("Ödev - 4 : Yoldaki Çizgileri Takip Etme", self)
        self.label_odev.setFont(bold_font)
        self.label_odev.setStyleSheet("color: #333; padding: 5px;")
        left_layout.addWidget(self.label_odev)
        
        self.label_id = QLabel("221229032", self)
        self.label_id.setFont(bold_font)
        self.label_id.setStyleSheet("color: #777; padding: 5px;")
        left_layout.addWidget(self.label_id)

        self.label_name = QLabel("Zehra Küçüker", self)
        self.label_name.setFont(bold_font)
        self.label_name.setStyleSheet("color: #555; padding: 5px;")
        left_layout.addWidget(self.label_name)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 500px; background-color: #f5f5f5;")
        self.image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        left_layout.addWidget(self.image_label)

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 300px; background-color: #f5f5f5;")
        self.processed_image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        right_layout.addWidget(self.processed_image_label)

        self.button = QPushButton("Görsel Yükle", self)
        self.button.setFont(QFont("Arial", 10))
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.button.clicked.connect(self.load_image)
        left_layout.addWidget(self.button)

        self.process_button = QPushButton("Çizgi Takip Et", self)
        self.process_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.process_button.clicked.connect(self.process_line_following)
        right_layout.addWidget(self.process_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        self.setGeometry(200, 200, 1000, 800)
        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Bir Görsel Seçin", "", "Tüm Dosyalar (*);;JPEG (*.jpg *.jpeg);;PNG (*.png)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def process_line_following(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

            edges = cv2.Canny(blurred, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=20)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            self.processed_image(img)

    def processed_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        h, w = img.shape[:2]
        q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.processed_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
        
class Odev5Page(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        bold_font = QFont("Arial", 14, QFont.Bold)

        self.label_odev = QLabel("Ödev - 5 : Göz Bulma", self)
        self.label_odev.setFont(bold_font)
        self.label_odev.setStyleSheet("color: #333; padding: 5px;")
        left_layout.addWidget(self.label_odev)
        
        self.label_id = QLabel("221229032", self)
        self.label_id.setFont(bold_font)
        self.label_id.setStyleSheet("color: #777; padding: 5px;")
        left_layout.addWidget(self.label_id)

        self.label_name = QLabel("Zehra Küçüker", self)
        self.label_name.setFont(bold_font)
        self.label_name.setStyleSheet("color: #555; padding: 5px;")
        left_layout.addWidget(self.label_name)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 500px; background-color: #f5f5f5;")
        self.image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        left_layout.addWidget(self.image_label)

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 300px; background-color: #f5f5f5;")
        self.processed_image_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        right_layout.addWidget(self.processed_image_label)

        self.button = QPushButton("Görsel Yükle", self)
        self.button.setFont(QFont("Arial", 10))
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.button.clicked.connect(self.load_image)
        left_layout.addWidget(self.button)

        self.detect_button = QPushButton("Gözleri Tespit Et", self)
        self.detect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.detect_button.clicked.connect(self.detect_eyes)
        right_layout.addWidget(self.detect_button)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        self.setGeometry(200, 200, 1000, 800)
        self.image_path = None

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Bir Görsel Seçin", "", "Tüm Dosyalar (*);;JPEG (*.jpg *.jpeg);;PNG (*.png)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def detect_eyes(self):
        if self.image_path:
            img = cv2.imread(self.image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Yüz tespiti için Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                
                # Yüz bölgesinde göz tespiti
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            self.processed_image(img)

    def processed_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        h, w = img.shape[:2]
        q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888 if len(img.shape) == 3 else QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.processed_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

class Odev6Page(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        bold_font = QFont("Arial", 14, QFont.Bold)

        self.label_odev = QLabel("Ödev - 6 : Nesne Sayma ve Özellik Çıkarma", self) 
        self.label_odev.setFont(bold_font)
        self.label_odev.setStyleSheet("color: #333; padding: 5px;")
        left_layout.addWidget(self.label_odev)

        self.label_id = QLabel("221229032", self)
        self.label_id.setFont(bold_font)
        self.label_id.setStyleSheet("color: #777; padding: 5px;")
        left_layout.addWidget(self.label_id)

        self.label_name = QLabel("Zehra Küçüker", self)
        self.label_name.setFont(bold_font)
        self.label_name.setStyleSheet("color: #555; padding: 5px;")
        left_layout.addWidget(self.label_name)

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 400px; background-color: #f5f5f5;")
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        self.button = QPushButton("Görsel Yükle", self)
        self.button.setFont(QFont("Arial", 10))
        self.button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.button.clicked.connect(self.load_image)
        left_layout.addWidget(self.button)

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 400px; background-color: #f5f5f5;")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.processed_image_label)

        self.process_button = QPushButton("Excel'e Dönüştür", self)
        self.process_button.setFont(QFont("Arial", 10))
        self.process_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        self.process_button.clicked.connect(self.process_image)
        right_layout.addWidget(self.process_button)
        
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Görseller (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio))

    def process_image(self):
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lower_green = np.array([0, 60, 0])
        upper_green = np.array([100, 200, 100])
        mask = cv2.inRange(img_rgb, lower_green, upper_green)

        labeled = label(mask)
        props = regionprops(labeled, intensity_image=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))

        data = []
        for i, prop in enumerate(props, start=1):
            y, x = prop.centroid
            length = prop.bbox[2] - prop.bbox[0]
            width = prop.bbox[3] - prop.bbox[1]
            diagonal = round(math.hypot(length, width), 2)
            energy = round(np.sum(prop.intensity_image[prop.image] ** 2), 2)
            entropy = round(shannon_entropy(prop.intensity_image[prop.image]), 2)
            mean = int(np.mean(prop.intensity_image[prop.image]))
            median = int(np.median(prop.intensity_image[prop.image]))

            data.append([
                i,
                f"{int(x)},{int(y)}",
                f"{length} px",
                f"{width} px",
                f"{diagonal} px",
                energy,
                entropy,
                mean,
                median
            ])

        df = pd.DataFrame(data, columns=["No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"])
        save_path, _ = QFileDialog.getSaveFileName(self, "Excel Dosyası Kaydet", "sonuc.xlsx", "Excel Dosyası (*.xlsx)")
        if save_path:
            df.to_excel(save_path, index=False)
            QMessageBox.information(self, "Başarılı", "Excel dosyası başarıyla oluşturuldu.")
 
        qimg_mask = QImage(mask.data, mask.shape[1], mask.shape[0], mask.strides[0], QImage.Format_Grayscale8)
        pixmap_mask = QPixmap.fromImage(qimg_mask)
        self.processed_image_label.setPixmap(pixmap_mask.scaled(self.processed_image_label.width(), self.processed_image_label.height(), Qt.KeepAspectRatio))

class MainPage(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        self.stacked_widget = QStackedWidget()
        self.buttons = []
        for i in range(1, 7):
            button = QPushButton(f"Ödev-{i}", self)
            button.setFont(QFont("Arial", 10))
            button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
            button.clicked.connect(self.change_page)
            self.buttons.append(button)
            top_layout.addWidget(button)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)
        self.setWindowTitle("Dijital Görüntü İşleme")
        self.setGeometry(200, 200, 1000, 800)

    def change_page(self):
        sender = self.sender()
        page_number = int(sender.text().split("-")[1]) - 1

        for button in self.buttons:
            button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        sender.setStyleSheet("background-color: #388E3C; color: white; padding: 10px; border-radius: 5px;")

        if self.stacked_widget.count() > page_number:
            self.stacked_widget.setCurrentIndex(page_number)
            return

        if page_number == 0:
            self.odev1_page = Odev1Page()
            self.stacked_widget.addWidget(self.odev1_page)
        elif page_number == 1:
            self.odev2_page = Odev2Page()
            self.stacked_widget.addWidget(self.odev2_page)
        elif page_number == 2:
            self.odev3_page = Odev3Page()
            self.stacked_widget.addWidget(self.odev3_page)
        elif page_number == 3:
            self.odev4_page = Odev4Page()
            self.stacked_widget.addWidget(self.odev4_page)
        elif page_number == 4:
            self.odev5_page = Odev5Page()
            self.stacked_widget.addWidget(self.odev5_page)
        elif page_number == 5:
            self.odev6_page = Odev6Page()
            self.stacked_widget.addWidget(self.odev6_page)
        else:
            empty_widget = QWidget()
            self.stacked_widget.addWidget(empty_widget)

        self.stacked_widget.setCurrentIndex(page_number)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainPage()
    main_win.show()
    sys.exit(app.exec_())
