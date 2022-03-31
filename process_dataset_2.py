import torch
from PIL import Image
import numpy
from load_images import ImagesLoader
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

"""
The process images class is designed to create augmentations for training data and to create
a batches during training, this class works with the Imagesloadere class. This class contains 
the get_training_batch function you should call during training. 

Parameters info
training_path   = String path to your floder with train images          type: str
testing_path    = String path to your floder with test images           type: str
validate_path   = String path to your floder with validate images       type: str
in_ress         = required input ressolution for model                  type: int
out_ress        = required output ressolution from model                type: int


aug_count = number which represents the number of 
dataset augmentations, number 5 mean you will get   
5x training data (rotated, changed brightness and 
contrast, flip).                                    type: int
"""

class Process_dataset:
    def __init__(self, in_ress, out_ress, training_path, aug_count, testing_path= False, validate_path= False):
        self.in_ress           = in_ress
        self.out_ress          = out_ress
        self.training_path     = training_path
        self.testing_path      = testing_path
        self.validate_path     = validate_path
        self.aug_count         = aug_count

        self.training_images   = []
        self.training_labels   = []
        self.validate_images   = []
        self.validate_labels   = []
        self.testing_images    = []
        self.testing_labels    = []

        if self.aug_count < 0:
            raise Exception("The number of augments aug_count can not be negative")

        train_loader = ImagesLoader(self.training_path, self.in_ress, self.out_ress)
        training_images_raw, training_labels_raw = train_loader.load_images()

        if self.testing_path:
            test_loader = ImagesLoader(self.testing_path, self.in_ress, self.out_ress)
            testing_images, testing_labels = test_loader.load_images()

            self.testing_images.append(testing_images)
            self.testing_labels.append(testing_labels)

            self.testing_count = len(testing_images)
            print("Testing images count - {}".format(self.testing_count))

        if self.validate_path:
            validate_loader = ImagesLoader(self.validate_path, self.in_ress, self.out_ress)
            validate_images, validate_labels = validate_loader.load_images()

            self.validate_images.append(validate_images)
            self.validate_labels.append(validate_labels)

            self.validate_count = len(validate_images)
            print("Validate images count - {}".format(self.validate_count))


        self.training_images.append(training_images_raw)
        self.training_labels.append(training_labels_raw)

        if self.aug_count !=0:
            self.training_images.append(training_images_raw)
            self.training_labels.append(training_labels_raw)

            images_aug,labels_aug = self._auqumentation(self.training_images, self.training_labels, self.aug_count)

            self.training_images.append(images_aug)
            self.training_labels.append(labels_aug)

            self.training_count = len(training_images_raw) * 2 + len(images_aug)
            print("Training images count - {}".format(self.training_count))
            print("\n")

        else:
            self.training_count = len(training_images_raw)
            print("Training images count - {}".format(self.training_count))
            print("\n")


    def get_training_batch(self, batch_size):
        return self.get_batch(self.training_images, self.training_labels, batch_size)

    def get_training_count(self):
        return self.training_count

    def get_testing_batch(self, batch_size):
        if self.testing_path:
            return self.get_batch(self.testing_images, self.testing_labels, batch_size, training=False)
        else:
            raise Exception("No testing data... Please specify training path folder in Process_dataset class")

    def get_testing_count(self):
        if self.testing_path:
            return self.testing_count
        else:
            raise Exception("No testing data... Please specify testing path folder in Process_dataset class")

    def get_validate_batch(self, batch_size):
        if self.validate_path:
            return self.get_batch(self.validate_images, self.validate_labels, batch_size, training=False)
        else:
            raise Exception("No validate data... Please specify validate path folder in Process_dataset class")

    def get_validate_count(self):
        if self.validate_path:
            return self.validate_count
        else:
            raise Exception("No validate data... Please specify validate path folder in Process_dataset class")


    def process_augumentation(self, image, label):
        image_aug = numpy.array((image), dtype=numpy.uint8)
        label_aug = numpy.array((label), dtype=numpy.uint8)

        angle_max = 25
        angle     = self._rnd(-angle_max, angle_max)

        image_in  = Image.fromarray(numpy.moveaxis(image_aug, 0, 2), 'RGB')
        mask_in   = Image.fromarray(numpy.moveaxis(label_aug, 0, 2), 'RGB')

        image_aug = image_in.rotate(angle,resample=Image.BICUBIC)
        mask_aug  = mask_in.rotate(angle,resample=Image.BICUBIC)

        image_aug = numpy.array(image_aug)
        mask_aug  = numpy.array(mask_aug)

        image_aug = numpy.swapaxes(image_aug, 0, 2)
        mask_aug  = numpy.swapaxes(mask_aug, 0, 2)

        return image_aug, mask_aug


    def _auqumentation(self, images, labels, aug_count):
        count      = len(images[0])
        counter    = 0
        images_aug = numpy.zeros((count * aug_count, 3, self.in_ress, self.in_ress), dtype=numpy.uint8)
        labels_aug = numpy.zeros((count * aug_count, 3, self.out_ress, self.out_ress), dtype=numpy.uint8)

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            results = [None] * count * aug_count
            for x in range(count * aug_count):
                results[x] = executor.submit(self.process_augumentation,images[0][counter],labels[0][counter])
                counter+=1
                if counter == count:
                    counter = 0

            counter = 0
            for f in concurrent.futures.as_completed(results):
                images_aug[counter], labels_aug[counter] = f.result()[0], f.result()[1]
                counter += 1

        return images_aug, labels_aug


    def get_batch(self,images, labels, batch_size, training = True):
        result_x = torch.zeros((batch_size, 3, self.in_ress, self.in_ress)).float()
        result_y = torch.zeros((batch_size, 3, self.out_ress, self.out_ress)).float()

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = [None] * batch_size
            for x in range(batch_size):
                results[x] = executor.submit(self.process_batch,images,labels,training)

            counter = 0
            for f in concurrent.futures.as_completed(results):
                result_x[counter], result_y[counter] = f.result()[0], f.result()[1]
                counter += 1

        return result_x, result_y


    def process_batch(self,images,labels, training):
        group_idx = numpy.random.randint(len(images))
        image_idx = numpy.random.randint(len(images[group_idx]))

        image_np  = numpy.array(images[group_idx][image_idx])/ 255.0
        label_np  = numpy.array(labels[group_idx][image_idx])/ 255.0

        if not training:
            result_x = torch.from_numpy(image_np).float()
            result_y = torch.from_numpy(label_np).float()

            return result_x, result_y

        elif self.aug_count == 0:
            result_x = torch.from_numpy(image_np).float()
            result_y = torch.from_numpy(label_np).float()

            return result_x, result_y

        """normalized_input = (label_np - numpy.amin(label_np)) / (numpy.amax(label_np) - numpy.amin(label_np))
        normalized_input = 2 * normalized_input - 1
        label_np = normalized_input

        normalized_input = (image_np - numpy.amin(image_np)) / (numpy.amax(image_np) - numpy.amin(image_np))
        normalized_input = 2 * normalized_input - 1
        image_np = normalized_input

        #image_np = 2 * image_np / 255. - 1
        #label_np = 2 * label_np / 255. - 1
        """

        if group_idx == 1:
            image_np, mask_np = self._augmentation_flip(image_np, label_np)

        else:
            image_np, mask_np = self._augmentation_flip(image_np, label_np)
            image_np, mask_np = self._augmentation_noise(image_np, mask_np)

        result_x = torch.from_numpy(image_np).float()
        result_y = torch.from_numpy(mask_np).float()

        return result_x, result_y

    def _augmentation_flip(self, image_np,label_np):
        if self._rnd(0, 1) < 0.5:
            aug_img   = numpy.flip(image_np, 1)
            aug_label = numpy.flip(label_np, 1)
        else:
            aug_img   = numpy.flip(image_np, 2)
            aug_label = numpy.flip(label_np, 2)

        return aug_img.copy(), aug_label.copy()

    def _augmentation_noise(self, image_np,label_np):
        brightness = self._rnd(-0.30, 0.25)
        contrast   = self._rnd(0.6, 1.1)
        #noise = 0.05 * (2.0 * numpy.random.rand(3, 384, 384) - 1.0)

        #noise_low = numpy.swapaxes(noise, 0, 2)
        #noise_low = cv2.resize(noise_low, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        #noise_low = numpy.swapaxes(noise_low, 0, 2)

        img_result = image_np + brightness
        img_result = 0.5 + contrast * (img_result - 0.5)
        #img_result = img_result + noise_low

        label_result = label_np + brightness
        label_result = 0.5 + contrast * (label_result - 0.5)
        #label_result = label_result + noise

        return numpy.clip(img_result, 0.0, 1.0),numpy.clip(label_result, 0.0, 1.0)


    def _rnd(self, min_value, max_value):
        return (max_value - min_value) * numpy.random.rand() + min_value