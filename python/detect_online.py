from ctypes import *
#import math
#import random
#import cv2
import os
# def sample(probs):
#     s = sum(probs)
#     probs = [a/s for a in probs]
#     r = random.uniform(0, 1)
#     for i in range(len(probs)):
#         r = r - probs[i]
#         if r <= 0:
#             return i
#     return len(probs)-1


class yolo_online_det():
    def __init__(self, yolo_exp):
        self._init_py_wrapper_stuff(yolo_exp.misc.proj_root_directory)
        self.classname_list = []
        with open(yolo_exp.class_names_file, 'r') as c_n_f:
            for classname in c_n_f.readlines():
                self.classname_list.append(classname.strip())
        self.network_cfg_file = yolo_exp.network_cfg_file
        self.weights_file_online = yolo_exp.weights_file_online
        self.detection_thresh_online = yolo_exp.detection_thresh_online
        self.nms_thresh_online = yolo_exp.nms_thresh_online
        self.set_gpu(yolo_exp.misc.gpu_id)
        self.net = self.load_net(self.network_cfg_file, self.weights_file_online, 0)


    def c_array(self, ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    class BOX(Structure):
        _fields_ = [("x", c_float),
                    ("y", c_float),
                    ("w", c_float),
                    ("h", c_float)]

    class IMAGE(Structure):
        _fields_ = [("w", c_int),
                    ("h", c_int),
                    ("c", c_int),
                    ("data", POINTER(c_float))]

    class METADATA(Structure):
        _fields_ = [("classes", c_int),
                    ("names", POINTER(c_char_p))]

    def _init_py_wrapper_stuff(self, project_root_dir):
        libdarknet_path = os.path.join(project_root_dir, 'dl_algos', 'darknet', 'libdarknet.so')
        self.lib = CDLL(libdarknet_path, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = self.IMAGE

        self.make_boxes = self.lib.make_boxes
        self.make_boxes.argtypes = [c_void_p]
        self.make_boxes.restype = POINTER(self.BOX)

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.num_boxes = self.lib.num_boxes
        self.num_boxes.argtypes = [c_void_p]
        self.num_boxes.restype = c_int

        self.make_probs = self.lib.make_probs
        self.make_probs.argtypes = [c_void_p]
        self.make_probs.restype = POINTER(POINTER(c_float))

        self.detect = self.lib.network_predict
        self.detect.argtypes = [c_void_p, self.IMAGE, c_float, c_float, c_float, POINTER(self.BOX), POINTER(POINTER(c_float))]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [self.IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [self.IMAGE, c_int, c_int]
        self.letterbox_image.restype = self.IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = self.METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = self.IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [self.IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, self.IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.network_detect = self.lib.network_detect
        self.network_detect.argtypes = [c_void_p, self.IMAGE, c_float, c_float, c_float, POINTER(self.BOX), POINTER(POINTER(c_float))]


        return

    def array_to_image(self, arr):
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = (arr/255.0).flatten()
        data = self.c_array(c_float, arr)
        im = self.IMAGE(w,h,c,data)
        return im

    def det(self, image):
        hier_thresh = 0.5
        image = self.array_to_image(image)
        self.rgbgr_image(image)

        boxes = self.make_boxes(self.net)
        probs = self.make_probs(self.net)
        num =   self.num_boxes(self.net)
        self.network_detect(self.net, image, self.detection_thresh_online, hier_thresh, self.nms_thresh_online, boxes, probs)
        all_dets = []
        for j in range(num):
            for i in range(len(self.classname_list)):
                if probs[j][i] > 0:
                    x1 = boxes[j].x - boxes[j].w/2.0
                    y1 = boxes[j].y - boxes[j].h/2.0
                    x2 = boxes[j].x + boxes[j].w/2.0
                    y2 = boxes[j].y + boxes[j].h/2.0
                    all_dets.append([self.classname_list[i], probs[j][i], [x1, y1, x2, y2]])
        all_dets = sorted(all_dets, key=lambda x: -x[1])
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return all_dets


    # def detect(self, im):
    #     #im = load_image(image, 0, 0)
    #     img = im.transpose(2, 0, 1)
    #     c, h, w = img.shape[0], img.shape[1], img.shape[2]
    #     img = (img/255.0).flatten()
    #     data = c_array(c_float, img)
    #     im = IMAGE(w, h, c, data)

    #     boxes = make_boxes(self.net)
    #     probs = make_probs(self.net)
    #     num =   num_boxes(self.net)
    #     hier_thresh = 0

    #     network_detect(self.net, im, self.detection_thresh_online, hier_thresh, self.nms_thresh_online, boxes, probs)

    #     all_dets = []
    #     for j in range(num):
    #         for i in range(len(self.classname_list)):
    #             if probs[j][i] > 0:
    #                 all_dets.append((i, probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))

    #     all_dets = sorted(all_dets, key=lambda x: -x[1])
    #     free_ptrs(cast(probs, POINTER(c_void_p)), num)

    #     return all_dets