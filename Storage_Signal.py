from PyQt5.QtCore import QObject, pyqtSignal

class Signal(QObject):
    # Tín hiệu truyền dữ liệu

    #Button

    #region Control
    click_Logo = pyqtSignal()
    click_B_Reset = pyqtSignal()
    click_B_Drag_Drop = pyqtSignal(bool)
    click_B_Pause_Run = pyqtSignal(bool)
    click_B_Switch  = pyqtSignal(int)
    click_B_Zero = pyqtSignal()
    click_B_Home = pyqtSignal()
    Send = pyqtSignal(str)
    #endregion
    
    #region Connect
    close_Screen = pyqtSignal()
    State_Com = pyqtSignal(bool)
    #endregion

    #region Current Position
    Data_Current_Position = pyqtSignal(list)
    #endregion

    #region Kinematics
    Data_FK_Theta1 = pyqtSignal(int)
    Data_FK_Theta2 = pyqtSignal(int)
    Data_FK_Theta3 = pyqtSignal(int)
    Data_FK_Theta = pyqtSignal(int)
    Data_IK = pyqtSignal(list)
    #endregion

    #region Manipilation
    Data_MP = pyqtSignal(str)
    Run_OK = pyqtSignal()
    #endregion

signal = Signal()
