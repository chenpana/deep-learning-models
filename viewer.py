#coding:utf-8
#try to add something to commit
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
from matplotlib.figure import Figure
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'viewer'))
    print(os.getcwd())
except:
    pass

#%%
import sys
import time
import numpy as np
import nibabel as nib
import tables
import scipy.ndimage.interpolation
import matplotlib.cm as cm
import matplotlib.colors as colors
import nilearn.image
import cv2
from functools import partial

from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

#%%
## https://blog.csdn.net/zhulove86/article/details/52563298
## https://matplotlib.org/gallery/user_interfaces/embedding_in_qt_sgskip.html
## https://doc.qt.io/qt-5/qwidget.html
## https://pythonspot.com/pyqt5-horizontal-layout/
## https://matplotlib.org/users/event_handling.html

#%%

def array_to_img(lower, high, array):
    return array 
    """
    img=np.zeros(array.shape+(3,),dtype=np.uint8)
    for j in range(array.shape[0]):
        for i in range(array.shape[1]):
            value = max(lower, array[j,i]) 
            value = min(high,value)
            img[j,i,:] = 255*(value - lower)/(high-lower)
    return img
    """

class Slicer:
    def __init__(self, data=None):
        if data is not None:
            self.load_img(data)
        self.label_color = tuple(color for name, color in colors.TABLEAU_COLORS.items())
        self.label_zoom = [] 
        self.label_value = []
        self.line_style = ('-', '--', '-.', ':',)
        self.line_width = (1,2)
        #self.dshow_rang = [-1000,2000]

    def load_img(self, data):
        # reorder to RAS coordinate system
        data_nii = nilearn.image.reorder_img(data)
        self.array_to_img=partial(array_to_img,data_nii.get_data().min(),data_nii.get_data().max())

        # equal xyz pixel spacing 
        affine = np.array(data_nii.affine)
        affine[2, 2] = affine[0, 0]
        data_nii = nilearn.image.resample_img(data_nii, affine)

        data = data_nii.get_data()
        self.dshow_rang=[np.percentile(data,1),np.percentile(data,99)]
        # transpose axis order from RAS(xyz) to numpy zyx 
        self.data_zoom = np.rot90(data, k=1, axes=(0, 2))
        self.data_indice = (np.array(self.data_zoom.shape) / 2).astype(np.int)

        # clear label
        self.label_zoom = [] 

    def load_label(self, data):
        # reorder to RAS coordinate system
        label_nii = nilearn.image.reorder_img(data)

        # zoom to img shape
        affine = np.array(label_nii.affine)
        affine[2, 2] = affine[0, 0]
        label_nii = nilearn.image.resample_img(
            label_nii,
            affine,
            target_shape=self.data_zoom.shape[::-1],
            interpolation='nearest')
        label_value = np.unique(label_nii.get_data().astype(np.uint8))[1:]
        label_value = np.concatenate([self.label_value, label_value])
        self.label_value = np.unique(label_value).astype(np.uint8)

        # transpose axis order from RAS(xyz) to numpy zyx 
        label_zoom = np.rot90(
            label_nii.get_data(), k=1, axes=(0, 2))
        self.label_zoom.append(label_zoom)

    def get_x_sec(self, x):
        x = int(x)
        img = self.data_zoom[:, :, x]
        return np.flip(img, 1)

    def get_y_sec(self, y):
        y = int(y)
        img = self.data_zoom[:, y, :]
        return img

    def get_z_sec(self, z):
        z = int(z)
        img = self.data_zoom[z, :, :]
        return np.flip(img, 0)

    def draw_label(self, axe, label, nth_label):
        # get label value
        label_value = self.label_value
        for i,value in enumerate(label_value):
            data = np.zeros(label.shape, dtype=np.uint8)
            data[label == value] = 1
            if np.sum(data) == 0:
                continue
            contours = cv2.findContours(data, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)[1]
            for c in contours:
                cont = np.reshape(c, (c.shape[0] * c.shape[-1]))
                x = cont[0::2]
                y = cont[1::2]
                axe.plot(x, y, color=self.label_color[i], lw=self.line_width[nth_label], ls=self.line_style[nth_label])
                axe.plot([x[-1], x[0]], [y[-1], y[0]],
                         color=self.label_color[i], lw=self.line_width[nth_label], ls=self.line_style[nth_label])

    def draw_z_sec(self, axe):
        arr = self.get_z_sec(self.data_indice[0])
        axe.imshow(arr, cmap=cm.gray, vmin=self.dshow_rang[0], vmax=self.dshow_rang[1])
        if len(self.label_zoom) <1:
            return
        #for label_zoom in self.label_zoom:
        for i,label_zoom in enumerate(self.label_zoom):
            label = np.flip(label_zoom[self.data_indice[0], :, :], 0)
            self.draw_label(axe, label, i)
        #axe.imshow(np.flip(self.label_zoom[self.data_indice[0],:,:],0),alpha=0.5)

    def draw_y_sec(self, axe):
        arr = self.get_y_sec(self.data_indice[1])
        axe.imshow(arr, cmap=cm.gray, vmin=self.dshow_rang[0], vmax=self.dshow_rang[1])
        if len(self.label_zoom) <1:
            return
        for i,label_zoom in enumerate(self.label_zoom):
            label = label_zoom[:, self.data_indice[1], :]
            self.draw_label(axe, label, i)
        #axe.imshow(self.label_zoom[:,self.data_indice[1],:],alpha=0.5)

    def draw_x_sec(self, axe):
        arr = self.get_x_sec(self.data_indice[2])
        axe.imshow(arr, cmap=cm.gray, vmin=self.dshow_rang[0], vmax=self.dshow_rang[1])
        if len(self.label_zoom) <1:
            return
        #for label_zoom in self.label_zoom:
        for i,label_zoom in enumerate(self.label_zoom):
            label = np.flip(label_zoom[:, :, self.data_indice[2]], 1)
            self.draw_label(axe, label, i)
        #axe.imshow(np.flip(self.label_zoom[:,:,self.data_indice[2]],1),alpha=0.5)

    def view_to_data_z(self, xy):
        self.data_indice[2] = xy[0]
        self.data_indice[1] = self.data_zoom.shape[1] - xy[1]
        return self.data_indice

    def view_to_data_y(self, xy):
        self.data_indice[0] = xy[1]
        self.data_indice[2] = xy[0]
        return self.data_indice

    def view_to_data_x(self, xy):
        self.data_indice[1] = self.data_zoom.shape[1] - xy[0]
        self.data_indice[0] = xy[1]
        return self.data_indice


#%%
class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.main_layout = QtWidgets.QHBoxLayout(self._main)
        self.init_left_space(self.main_layout)
        self.init_middle_space(self.main_layout)
        self.init_panel_space(self.main_layout)
        self.init_data()
        self.cmap = cm.gray

    def init_data(self):
        self.data_list = []

    def init_left_space(self, layout):
        self.axis_can = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.axis_can)
        #self.addToolBar(NavigationToolbar(static_canvas, self))

        self.axis_axe = self.axis_can.figure.subplots()
        #t = np.linspace(0, 10, 501)
        #self.axis_axe.plot(t, np.tan(t), ".")

        self.axis_can.mpl_connect('button_press_event', self.on_axis_clicked)
        self.axis_can.mpl_connect('scroll_event', self.on_axis_scroll)

    def on_axis_clicked(self, event):
        self.current_slicer.view_to_data_z((event.xdata, event.ydata))
        self.update_cron()
        self.update_sagi()

    def on_cron_clicked(self, event):
        self.current_slicer.view_to_data_y((event.xdata, event.ydata))
        self.update_axis()
        self.update_sagi()

    def on_sagi_clicked(self, event):
        self.current_slicer.view_to_data_x((event.xdata, event.ydata))
        self.update_axis()
        self.update_cron()

    def on_axis_scroll(self, event):
        if event.button == 'up':
            self.current_slicer.data_indice[0] += 1
        else:
            self.current_slicer.data_indice[0] -= 1
        self.update_axis()

    def on_cron_scroll(self, event):
        if event.button == 'up':
            self.current_slicer.data_indice[1] += 1
        else:
            self.current_slicer.data_indice[1] -= 1
        self.update_cron()

    def on_sagi_scroll(self, event):
        if event.button == 'up':
            self.current_slicer.data_indice[2] += 1
        else:
            self.current_slicer.data_indice[2] -= 1
        self.update_sagi()

    def init_middle_space(self, layout):
        self.middle_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.middle_layout)

        self.cron_can = FigureCanvas(Figure(figsize=(5, 3)))
        self.cron_axe = self.cron_can.figure.subplots()
        #self.cron_axe.plot(t, np.tan(t), ".")
        #t = np.linspace(0, 10, 501)
        self.middle_layout.addWidget(self.cron_can)

        self.sagi_can = FigureCanvas(Figure(figsize=(5, 3)))
        self.sagi_axe = self.sagi_can.figure.subplots()
        #t = np.linspace(0, 10, 501)
        #self.sagi_axe.plot(t, np.tan(t), ".")

        self.middle_layout.addWidget(self.sagi_can)

        self.cron_can.mpl_connect('button_press_event', self.on_cron_clicked)
        self.cron_can.mpl_connect('scroll_event', self.on_cron_scroll)
        self.sagi_can.mpl_connect('button_press_event', self.on_sagi_clicked)
        self.sagi_can.mpl_connect('scroll_event', self.on_sagi_scroll)

    def init_panel_space(self, layout):
        topWidget = QtWidgets.QWidget() 
        topWidget.setFixedWidth(100)
        layout.addWidget(topWidget)
        self.panel_layout = QtWidgets.QVBoxLayout(topWidget)

        load_img_btn = QtWidgets.QPushButton('Load NII')
        self.panel_layout.addWidget(load_img_btn,1)
        load_img_btn.clicked.connect(self.on_img_btn_clicked)

        load_label_btn = QtWidgets.QPushButton('Load Label')
        self.panel_layout.addWidget(load_label_btn,1)
        load_label_btn.clicked.connect(self.on_lbl_btn_clicked)

        #self.addToolBar(QtCore.Qt.BottomToolBarArea,
        #                NavigationToolbar(dynamic_canvas, self))

    def on_img_btn_clicked(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "QFileDialog.getOpenFileNames()",
            "",
            "All Files (*);;Python Files (*.py)",
            options=options)
        if files is None or len(files) < 1:
            return
        print(files)
        data_nii = nib.load(files[0])
        self.current_slicer = Slicer(data_nii)

        self.update_axis()
        self.update_sagi()
        self.update_cron()

    def on_lbl_btn_clicked(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", "",
            "All Files (*);;NII Files (*.nii);(*.nii.gz)")
        if files is None or len(files) < 1:
            return
        print(files)
        data_nii = nib.load(files[0])
        self.current_slicer.load_label(data_nii)

        self.update_axis()
        self.update_sagi()
        self.update_cron()

    def update_axis(self):
        self.axis_axe.clear()
        self.current_slicer.draw_z_sec(self.axis_axe)
        self.axis_axe.figure.canvas.draw()

    def update_cron(self):
        self.cron_axe.clear()
        self.current_slicer.draw_y_sec(self.cron_axe)
        self.cron_axe.figure.canvas.draw()

    def update_sagi(self):
        self.sagi_axe.clear()
        self.current_slicer.draw_x_sec(self.sagi_axe)
        self.sagi_axe.figure.canvas.draw()


#%%
if __name__ == "__main__":
    qapp = QtWidgets.QApplication(sys.argv)
    app = ApplicationWindow()
    app.show()
    qapp.exec_()
