import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainWidget import Ui_MainWindow
from lidaUiManager import LidaUiManager

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    manager = LidaUiManager(ui, MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
