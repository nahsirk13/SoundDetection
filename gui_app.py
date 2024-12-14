import sys

import joblib
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QHBoxLayout, QSpacerItem, QSizePolicy, QGridLayout, QTabWidget, QLineEdit,
    QTableWidget, QTableWidgetItem, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon
import threading

from scipy.signal import spectrogram

import audio_processor
import sound_spectrum_analysis
from database import init_db, save_result, query_results
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from extract_features import extract_features


class SoundClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(QIcon("logo.png"))
        # Load the trained model
        try:
            self.model = joblib.load("sound_classifier.pkl")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

        self.running = False
        self.direction_mode_running = False
        self.last_sounds = ["None", "None", "None"]  # Track last three sounds

        init_db()  # Initialize the database

    def initUI(self):
        self.setWindowTitle("Sound Classifier with History")
        self.setGeometry(100, 100, 800, 700)
        self.setStyleSheet("background-color: #2E3440; color: #D8DEE9;")

        # Tab layout
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Main Tab
        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Main")
        self.setup_main_tab()

        # History Tab
        self.history_tab = QWidget()
        self.tabs.addTab(self.history_tab, "History")
        self.setup_history_tab()

    def setup_main_tab(self):
        layout = QVBoxLayout()

        # Logo
        logo = QLabel(self)
        pixmap = QPixmap("logo.png")
        pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        # Title
        title_label = QLabel("Sound Classifier")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Helvetica", 24, QFont.Bold))
        title_label.setStyleSheet("color: #88C0D0; margin: 10px;")
        layout.addWidget(title_label)

        # Status Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Helvetica", 16))
        self.status_label.setStyleSheet("color: #A3BE8C; margin: 10px;")
        layout.addWidget(self.status_label)

        # Result Label
        self.result_label = QLabel("Result: None")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Helvetica", 16, QFont.Bold))
        self.result_label.setStyleSheet("color: #EBCB8B; margin: 10px;")
        layout.addWidget(self.result_label)

        # Last Sounds Layout
        last_sounds_layout = QGridLayout()
        self.just_now_label = QLabel("Just Now: None")
        self.just_now_label.setFont(QFont("Helvetica", 14))
        self.just_now_label.setStyleSheet("color: #88C0D0; margin: 5px;")
        last_sounds_layout.addWidget(self.just_now_label, 0, 0)

        self.five_seconds_ago_label = QLabel("~5 Seconds Ago: None")
        self.five_seconds_ago_label.setFont(QFont("Helvetica", 14))
        self.five_seconds_ago_label.setStyleSheet("color: #88C0D0; margin: 5px;")
        last_sounds_layout.addWidget(self.five_seconds_ago_label, 1, 0)

        self.ten_seconds_ago_label = QLabel("~10 Seconds Ago: None")
        self.ten_seconds_ago_label.setFont(QFont("Helvetica", 14))
        self.ten_seconds_ago_label.setStyleSheet("color: #88C0D0; margin: 5px;")
        last_sounds_layout.addWidget(self.ten_seconds_ago_label, 2, 0)
        layout.addLayout(last_sounds_layout)

        # Spectrogram Display
        self.spectrogram_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.spectrogram_axes = self.spectrogram_canvas.figure.add_subplot(111)
        layout.addWidget(self.spectrogram_canvas)

        # Buttons Layout
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.direction_button = QPushButton("Direction Mode")
        self.direction_button.clicked.connect(self.start_direction_mode)
        button_layout.addWidget(self.direction_button)

        self.exit_direction_button = QPushButton("Exit Direction Mode")
        self.exit_direction_button.clicked.connect(self.exit_direction_mode)
        self.exit_direction_button.setEnabled(False)
        button_layout.addWidget(self.exit_direction_button)

        layout.addLayout(button_layout)
        self.main_tab.setLayout(layout)

    def setup_history_tab(self):
        layout = QVBoxLayout()

        # Filters
        filter_layout = QHBoxLayout()
        self.sound_filter = QComboBox()
        self.sound_filter.addItems(["All", "Dog Bark", "Microwave", "Smoke Alarm", "Other"])
        filter_layout.addWidget(QLabel("Sound:"))
        filter_layout.addWidget(self.sound_filter)

        self.month_filter = QComboBox()
        self.month_filter.addItems([
            "All Months", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ])
        filter_layout.addWidget(QLabel("Month:"))
        filter_layout.addWidget(self.month_filter)

        self.filter_button = QPushButton("Search")
        self.filter_button.clicked.connect(self.search_history)
        filter_layout.addWidget(self.filter_button)

        layout.addLayout(filter_layout)

        # Results Table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Sound", "Timestamp"])
        layout.addWidget(self.results_table)

        self.history_tab.setLayout(layout)

    def update_status(self, status_text, color="#A3BE8C"):
        self.status_label.setText(f"Status: {status_text}")
        self.status_label.setStyleSheet(f"color: {color};")

    def update_last_sounds(self, new_sound):
        if " - " not in new_sound:
            new_sound = f"{new_sound} - direction unknown"
        self.last_sounds = [new_sound] + self.last_sounds[:2]
        self.just_now_label.setText(f"Just Now: {self.last_sounds[0]}")
        self.five_seconds_ago_label.setText(f"~5 Seconds Ago: {self.last_sounds[1]}")
        self.ten_seconds_ago_label.setText(f"~10 Seconds Ago: {self.last_sounds[2]}")

    def display_spectrogram(self, spectrogram):
        self.spectrogram_axes.clear()
        self.spectrogram_axes.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        self.spectrogram_axes.set_title("Spectrogram")
        self.spectrogram_canvas.draw()

    def start_detection(self):
        self.running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        threading.Thread(target=self.simple_identification).start()

    def stop_detection(self):
        self.running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Idle", "#BF616A")

    def simple_identification(self):
        if self.model is None:
            print("No model loaded. Classification cannot proceed.")
            self.result_label.setText("Result: Model not loaded")
            self.update_status("Idle", "#BF616A")
            return

        while self.running:
            self.update_status("Listening...", "#5E81AC")

            # Record audio and extract features
            file_path = audio_processor.record_audio_to_file()
            features = extract_features(file_path)

            if features is not None:
                try:
                    features = features.reshape(1, -1)  # Reshape for model input
                    result = self.model.predict(features) # Predict the sound
                    self.result_label.setText(f"Result: {result}")

                    # Save result to the database (optional)
                    save_result(result)
                    self.update_last_sounds(result)
                except Exception as e:
                    print(f"Error during classification: {e}")
                    self.result_label.setText("Result: Unable to classify")
            else:
                self.result_label.setText("Result: Unable to classify")

            self.update_status("Idle", "#A3BE8C")

    def start_direction_mode(self):
        self.direction_mode_running = True
        self.update_status("Processing Direction...", "#EBCB8B")
        self.direction_button.setEnabled(False)
        self.exit_direction_button.setEnabled(True)
        threading.Thread(target=self.direction_mode).start()

    def exit_direction_mode(self):
        self.direction_mode_running = False
        self.update_status("Idle", "#A3BE8C")
        self.direction_button.setEnabled(True)
        self.exit_direction_button.setEnabled(False)

    def direction_mode(self):
        while self.direction_mode_running:
            audio_file = audio_processor.record_audio()
            if sound_spectrum_analysis.microwave_different_height(audio_file) == 1:
                dResult= "Different Height (Doppler Shift Found)"
            if sound_spectrum_analysis.microwave_different_height(audio_file) == -1:
                dResult= "Same Height (Doppler Shift Not Found)"

            self.result_label.setText(f"Result: {dResult}")
            self.update_last_sounds(dResult)
            #self.display_spectrogram(spectrogram)
            QApplication.processEvents()
            if not self.direction_mode_running:
                break

    def search_history(self):
        sound = self.sound_filter.currentText()
        sound = None if sound == "All" else sound

        # Get selected month
        selected_month = self.month_filter.currentText()
        month_mapping = {
            "January": "01", "February": "02", "March": "03", "April": "04",
            "May": "05", "June": "06", "July": "07", "August": "08",
            "September": "09", "October": "10", "November": "11", "December": "12"
        }
        month = month_mapping.get(selected_month) if selected_month != "All Months" else None

        # Query the database
        results = query_results(sound=sound, month=month)
        self.results_table.setRowCount(len(results))

        for row_idx, (sound, timestamp) in enumerate(results):
            self.results_table.setItem(row_idx, 0, QTableWidgetItem(sound))
            self.results_table.setItem(row_idx, 1, QTableWidgetItem(timestamp))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = SoundClassifierApp()
    main_window.show()
    sys.exit(app.exec_())
