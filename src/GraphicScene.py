from PyQt5.QtGui import QMovie

from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import QPointF, Qt, QSize


class GraphicScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.pixmap_item = QGraphicsPixmapItem()
        self.addItem(self.pixmap_item)


class GraphicsItem(QGraphicsPixmapItem):
    def __init__(self, parent=None, mainwindow=None):
        super().__init__(parent)
        self.mainwindow = mainwindow
        self.parent = parent
        self.image = None

    def center_item_in_scene(self):
        scene_rect = self.parent.sceneRect()
        item_rect = self.boundingRect()
        center_pos = QPointF(scene_rect.center().x() - item_rect.width() / 2,
                             scene_rect.center().y() - item_rect.height() / 2)
        self.setPos(center_pos)


    def paint(self, painter, option, widget):
        if self.image is not None:
            painter.drawImage(self.image.rect(), self.image)
            self.update()

    def setPixmap(self, pixmap):
        self.image = pixmap.toImage()
        super().setPixmap(pixmap)


class LoadingScene(QGraphicsScene):
    def __init__(self, gif_path, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(Qt.transparent)

        # Создаем QMovie для загрузки GIF
        self.movie = QMovie(gif_path)
        self.movie.setScaledSize(QSize(100, 100))

        self.pixmap_item = QGraphicsPixmapItem()
        self.addItem(self.pixmap_item)
        self.pixmap_item.setPos(
            self.width() / 2 - self.pixmap_item.boundingRect().width() / 2,
            self.height() / 2 - self.pixmap_item.boundingRect().height() / 2
        )

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

