import math
import os
from PIL import Image
import numpy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class ImagesLoader:
    """
    Images loader class is used to read images from a folder and to transform them into a numpy matrix,
    this class automatically creates images and labels for the network from high-resolution images 

    Parameters info
    Path = path to your floder with images             type: str
    in_ress = required input ressolution for model     type: int
    out_ress = required output ressolution from model  type: int

    """
    def __init__(self, path,in_ress,out_ress):
        self.path       = path
        self.in_ress    = in_ress
        self.out_ress   = out_ress
        self.channels   = 3

        self.files      = os.listdir(path)
        self.imgs_count = len(self.files)
        self.images     = numpy.zeros((self.imgs_count, self.channels, self.in_ress, self.in_ress), dtype=numpy.uint8)
        self.labels     = numpy.zeros((self.imgs_count, self.channels, self.out_ress, self.out_ress), dtype=numpy.uint8)


    def load_images(self):

        processes_count = multiprocessing.cpu_count()
        if self.imgs_count < processes_count:
            raise Exception("The number of images in the dataset is less than the number of CPU cores,"
                            " please increase the count of training images to at least {}".format(processes_count))

        num_for_process = math.floor(int(self.imgs_count) / processes_count)

        stop_start_ids = [[0 for _ in range(2)] for _ in range(processes_count)]

        for z in range(processes_count):
            if z == 0:
                stop_start_ids[z][1] = num_for_process

            if z == processes_count -1:
                stop_start_ids[z][0] = stop_start_ids[z - 1][1]
                stop_start_ids[z][1] = self.imgs_count
                continue

            stop_start_ids[z][0] = stop_start_ids[z - 1][1]
            stop_start_ids[z][1] = stop_start_ids[z][0] + num_for_process


        with ThreadPoolExecutor(max_workers= processes_count) as executor:
            results = [None] * processes_count
            for x in range(processes_count):
                results[x] = executor.submit(self.run_process, stop_start_ids[x])


        return self.images, self.labels


    def run_process(self, start_stop):

        counter = 0
        for i in range(start_stop[0], start_stop[1]):
            y = Image.open(os.path.join(self.path, self.files[i])).convert("RGB")
            #x = ImageOps.exif_transpose(x)

            if y.size[0] > y.size[1]:
                new_size = (y.size[0] / y.size[1]) * self.out_ress
                y = y.resize((math.floor(new_size), self.out_ress),Image.BICUBIC)

                y = y.crop((math.floor((y.size[0] - self.out_ress) / 2),
                            0, math.floor((y.size[0] - self.out_ress) / 2) + self.out_ress, self.out_ress))

            elif y.size[0] == y.size[1]:
                y = y.resize((self.out_ress, self.out_ress),Image.BICUBIC)

            else:
                new_size = (y.size[1] / y.size[0]) * self.out_ress
                y = y.resize((self.out_ress, math.floor(new_size)),Image.BICUBIC)

                y = y.crop((math.floor((y.size[0] - self.out_ress) / 2),
                            0, math.floor((y.size[0] - self.out_ress) / 2) + self.out_ress, self.out_ress))

            y_np = numpy.array(y)
            y_np = numpy.moveaxis(y_np,-1,0)

            self.labels[i]  = y_np

            x = y.resize((self.in_ress,self.in_ress),Image.BICUBIC)
            x_np = numpy.array(x)
            x_np = numpy.moveaxis(x_np,-1,0)

            self.images[i]  = x_np

            counter+= 1
