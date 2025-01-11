from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton,QMessageBox, QStackedWidget
import sys
from CurrentPosition import Ui_Current_Position
from Kinematics import Ui_Kinematics
from Header import Ui_Header
from Image_Processing import Ui_Image_Processing
from Manipilation_Planning import Ui_Manipilation_Planning
from Storage_Signal import signal

#data = Event()
""" Lưu trữ Signal và Slots của các sự kiện trong Home 
"""

class Home(QMainWindow):
    def __init__(self):
        super().__init__()

        signal.click_Logo.connect(self.open_windows_background)

        #region Set up Widget
        self.centralwidget = QWidget(self)
        self.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)
        self.resize(1400, 670) 
        self.setWindowTitle("Robotics")
        self.setWindowIcon(QtGui.QIcon("picture/logo_ute.png"))
        self.setStyleSheet("background-color: rgb(220, 220, 220);")
        
        # Tạo một QStackedWidget
        
        self.Manipilation_Planning = Ui_Manipilation_Planning()
        self.Manipilation_Planning.setupUi(self)
        self.Manipilation_Planning.F_Manipilation_Planning.hide()

        self.Kinematics = Ui_Kinematics()
        self.Kinematics.setupUi(self)
        #self.Kinematics.F_Kinematics.hide()                          

        self.Image_Processing = Ui_Image_Processing()
        self.Image_Processing.setupUi(self)
        self.Image_Processing.F_Image_Processing.hide()

        self.CurrentPosition = Ui_Current_Position()
        self.CurrentPosition.setupUi(self)

        self.Header = Ui_Header()
        self.Header.setupUi(self)

        #endregion
        signal.click_B_Switch.connect(self.Switch_Screen)
        signal.State_Com.connect(self.Read_Current_Position)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.Header.Update_Current_Position)  # Kết nối sự kiện timeout với hàm scan
        self.timer.stop()  # Thiết lập thời gian quét mỗi 50ms

    def Switch_Screen(self, data):
        if data == 0:
            self.Kinematics.F_Kinematics.show()     
            self.Image_Processing.F_Image_Processing.hide()
            self.Manipilation_Planning.F_Manipilation_Planning.hide()
        elif data == 1:
            self.Kinematics.F_Kinematics.hide()     
            self.Image_Processing.F_Image_Processing.show()
            self.Manipilation_Planning.F_Manipilation_Planning.hide()
        else:
            self.Kinematics.F_Kinematics.hide()     
            self.Image_Processing.F_Image_Processing.hide()
            self.Manipilation_Planning.F_Manipilation_Planning.show()

    def Read_Current_Position(self, data: bool):
        if data:
            self.timer.start(100)
        else:
            self.timer.stop()

    def open_windows_background(self):
        self.Backround_windows = BackGround()
        self.Backround_windows.show()
        self.close()  # Đóng cửa sổ hiện tại
    
    def closeEvent(self, event):
        #signal.close_Screen.emit()
        None

class BackGround(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Background")
        self.resize(1162, 730)


        self.image_background = QLabel(self)
        self.image_background.setPixmap(QtGui.QPixmap("picture/background.png"))
        self.image_background.setFixedSize(1162, 650)
        # Tạo nút để chuyển sang cửa sổ 1
        self.B_Swich = QPushButton("START", self)
        self.B_Swich.setGeometry(481, 680, 200, 40)  # Vị trí và kích thước của nút
        self.B_Swich.setStyleSheet("""
            QPushButton{
                font: 30px Bold "Century";
                border: 1px solid;
                border-radius: 10px;
                background-color: #27CA43;
            }               
        """)
        self.B_Swich.clicked.connect(self.open_window_home)

    def open_window_home(self):
        self.Home_windows = Home()
        self.Home_windows.show()
        self.close()  # Đóng cửa sổ hiện tại



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Home()
    mainWindow.show()
    sys.exit(app.exec_())