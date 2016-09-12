#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib._cntr import Cntr
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys

class testConfiguration(QWidget):
    def __init__(self, parent = None):
        super(testConfiguration, self).__init__(parent)

        # initiate Qt layout        
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("testing")
        
        # initiate labels
        # upper red, lower red
        self.l1 = QLabel("upper red")
        self.l1.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l1)       
        self.l2 = QLabel("lower red")
        self.l2.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l2)
        
        # blue range
        self.l3 = QLabel("upper blue")
        self.l3.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l3)        
        self.l4 = QLabel("lower blue")
        self.l4.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l4)
        
        
        # green range
        self.l5 = QLabel("upper green")
        self.l5.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l5)        
        self.l6 = QLabel("lower green")
        self.l6.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l6)
        
        self.l7 = QLabel("max Value")
        self.l7.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l7, 0, 3)
        self.l8 = QLabel("Threshold block size")
        self.l8.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l8, 1, 3)
        self.l9 = QLabel("Constant")
        self.l9.setAlignment(Qt.AlignLeading)
        self.layout.addWidget(self.l9, 2, 3)
        
        # initiate sliders
        self.upper_red = self.sliderMaker(0, 255, 255)
        self.lower_red = self.sliderMaker(0, 255, 150)
        self.upper_blue = self.sliderMaker(0, 255, 80)
        self.lower_blue = self.sliderMaker(0, 255, 0)
        self.upper_green = self.sliderMaker(0, 255, 150)
        self.lower_green = self.sliderMaker(0, 255, 20)
        self.max_value = self.sliderMaker(0, 255, 255)
        self.threshold_block_size = self.sliderMaker(3, 27, 11, interval= 2)
        self.constant_sub = self.sliderMaker(0, 100, 2)
        
        # add slider to layout  
        self.layout.addWidget(self.upper_red, 0, 2)
        self.layout.addWidget(self.lower_red, 1, 2)
        self.layout.addWidget(self.upper_blue, 2, 2)
        self.layout.addWidget(self.lower_blue, 3, 2)
        self.layout.addWidget(self.upper_green, 4, 2)
        self.layout.addWidget(self.lower_green, 5, 2)
        self.layout.addWidget(self.max_value, 0, 5)
        self.layout.addWidget(self.threshold_block_size, 1, 5)
        self.layout.addWidget(self.constant_sub, 2, 5)
        
        # slider signals
        self.upper_red.valueChanged.connect(self.changeValue)
        self.lower_red.valueChanged.connect(self.changeValue)
        self.upper_blue.valueChanged.connect(self.changeValue)
        self.lower_blue.valueChanged.connect(self.changeValue)
        self.upper_green.valueChanged.connect(self.changeValue)
        self.lower_green.valueChanged.connect(self.changeValue)
        self.max_value.valueChanged.connect(self.changeValue)
        self.threshold_block_size.valueChanged.connect(self.changeValue)
        self.constant_sub.valueChanged.connect(self.changeValue)
        
        # show values of slider bars
        self.l1_val = QLabel("%d" % self.upper_red.value())
        self.l2_val = QLabel("%d" % self.lower_red.value())
        self.l3_val = QLabel("%d" % self.upper_blue.value())
        self.l4_val = QLabel("%d" % self.lower_blue.value())
        self.l5_val = QLabel("%d" % self.upper_green.value())
        self.l6_val = QLabel("%d" % self.lower_green.value())
        self.l7_val = QLabel("%d" % self.max_value.value())
        self.l8_val = QLabel("%d" % self.threshold_block_size.value())
        self.l9_val = QLabel("%d" % self.constant_sub.value())
        
        # add values to layout
        self.layout.addWidget(self.l1_val, 0, 1)
        self.layout.addWidget(self.l2_val, 1, 1)
        self.layout.addWidget(self.l3_val, 2, 1)
        self.layout.addWidget(self.l4_val, 3, 1)
        self.layout.addWidget(self.l5_val, 4, 1)
        self.layout.addWidget(self.l6_val, 5, 1)
        self.layout.addWidget(self.l7_val, 0, 4)
        self.layout.addWidget(self.l8_val, 1, 4)
        self.layout.addWidget(self.l9_val, 2, 4)
        
    def changeValue(self):
        print("value changed")
        self.l1_val.setText("%d" % self.upper_red.value())
        self.l2_val.setText("%d" % self.lower_red.value())
        self.l3_val.setText("%d" % self.upper_blue.value())
        self.l4_val.setText("%d" % self.lower_blue.value())
        self.l5_val.setText("%d" % self.upper_green.value())
        self.l6_val.setText("%d" % self.lower_green.value())
        self.l7_val.setText("%d" % self.max_value.value())
        self.l8_val.setText("%d" % self.threshold_block_size.value())
        self.l9_val.setText("%d" % self.constant_sub.value())

    def sliderMaker(self, min_val, max_val, set_val, interval = 1):
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(min_val)
        sl.setMaximum(max_val)
        sl.setTickPosition(QSlider.TicksBelow)
        sl.setTickInterval(interval)
        sl.setValue(set_val)
        
        return sl
    
    def addImage(self, image):
        l_img = QLabel()
        l_img.setPixmap(QPixmap(image))
        self.layout.addWidget(l_img)

def testGUI():
    app = QApplication(sys.argv)
    ex = testConfiguration()
    ex.show()
    sys.exit(app.exec_())
    
    
# ============================ end of class and functions ======================