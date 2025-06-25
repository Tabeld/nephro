import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QRectF, Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QMovie
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from GraphicScene import GraphicScene, GraphicsItem, LoadingScene
from diagnostic import diagnostic


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.scene_size = None
        uic.loadUi('mainwindow.ui', self)
        self.scene = GraphicScene(self)
        self.loading_scene = LoadingScene("image\\loading.gif")
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setMouseTracking(True)
        self.pixmap_item = GraphicsItem(mainwindow=self)
        self.scene.addItem(self.pixmap_item)
        self.graphicsView.setRenderHint(QPainter.Antialiasing)
        self.label_2.setVisible(False)
        self.dir_name = None
        self.predictions = None
        self.current_image_index = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.event_initialization()

    def event_initialization(self):
        self.pushButton.clicked.connect(self.start_diagnostic)
        self.pushButton_2.clicked.connect(self.select_research)

    def select_research(self):
        self.dir_name = QFileDialog.getExistingDirectory(self)
        self.lineEdit.setText(self.dir_name)

    def start_diagnostic(self):
        if not self.dir_name:
            return

        self.graphicsView.setScene(self.loading_scene)
        self.loading_scene.start_animation()
        QtWidgets.QApplication.processEvents()

        self.worker = Worker(self.dir_name)
        self.worker.finished.connect(self.on_diagnostic_finished)
        self.worker.start()

    def on_diagnostic_finished(self, predictions):
        self.graphicsView.setScene(self.scene)
        print(len(predictions))
        self.predictions = predictions
        if len(self.predictions) != 0:
            self.current_image_index = 0
            self.load_image(self.current_image_index)

    def load_image(self, index):
        self.current_image_index = index
        image = self.predictions[self.current_image_index]
        print(image)
        height, width, _ = image.shape
        print(height, width)
        bytes_per_line = 4 * width
        q_image = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGBA8888
        )

        self.scene_size = QRectF(0, 0, width, height)
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)

    def reset_scene(self):
        self.scene.setSceneRect(self.scene_size)
        self.center_item()

    def center_item(self):
        scene_rect = self.scene.sceneRect()
        item_rect = self.pixmap_item.boundingRect()
        center_x = (scene_rect.width() - item_rect.width()) / 2
        center_y = (scene_rect.height() - item_rect.height()) / 2
        self.pixmap_item.setPos(center_x, center_y)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Right:
            self.next_image()
        elif e.key() == Qt.Key_Left:
            self.previous_image()

    def next_image(self):
        if self.current_image_index + 1 < len(self.predictions):
            self.load_image(self.current_image_index + 1)

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.load_image(self.current_image_index - 1)

    class LoadingScene(QGraphicsScene):
        def __init__(self, gif_path, parent=None):
            super().__init__(parent)
            self.setBackgroundBrush(Qt.black)  # Черный фон

            # Создаем QMovie для загрузки GIF
            self.movie = QMovie(gif_path)
            self.movie.setScaledSize(QSize(100, 100))  # Масштабируем GIF

            # Создаем элемент для отображения кадров GIF
            self.pixmap_item = QGraphicsPixmapItem()
            self.addItem(self.pixmap_item)
            self.pixmap_item.setPos(
                self.width() / 2 - self.pixmap_item.boundingRect().width() / 2,
                self.height() / 2 - self.pixmap_item.boundingRect().height() / 2
            )

            # Подключаем сигнал изменения кадра
            self.movie.frameChanged.connect(self.update_pixmap)

        def start_animation(self):
            """Запуск анимации"""
            self.movie.start()

        def stop_animation(self):
            """Остановка анимации"""
            self.movie.stop()

        def update_pixmap(self):
            """Обновление кадра анимации"""
            current_pixmap = self.movie.currentPixmap()
            self.pixmap_item.setPixmap(current_pixmap)

            # Центрируем элемент при изменении размера
            self.pixmap_item.setPos(
                self.width() / 2 - current_pixmap.width() / 2,
                self.height() / 2 - current_pixmap.height() / 2
            )

class Worker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, dir_name):
        super().__init__()
        self.dir_name = dir_name

    def run(self):
        predictions = diagnostic(self.dir_name)
        self.finished.emit(predictions)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
