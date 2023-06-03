'''
create a dataset and dataloader,laod mnist data
1.Extract the training images, training labels, test images, and test labels from the dataset.
2.Define a dataset class that takes the input images and labels.
3.Define a dataloader that outputs a random batch of data with the desired batch size.
4.Verify the correctness of the dataset and dataloader.
'''
import matplotlib.pyplot as plt
import numpy as np


class MNISTDataset:
    def __init__(self, images_file, labels_file) -> None:
        self.images = self.parse_mnist_images_file(images_file)
        self.labels = self.parse_mnist_labels_file(labels_file)
        assert len(self.images) == len(
            self.labels), "Create mnist dataset failed"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    @staticmethod
    def parse_mnist_labels_file(file_path):
        # return the labels
        with open(file_path, "rb") as f:
            data = f.read()
        # magic_number,number of items
        magic_number, number_of_items = np.frombuffer(
            data, dtype=">i", count=2, offset=0)

        # magic number is 2049
        assert magic_number == 2049, "Invalid labels files"
        items = np.frombuffer(data, dtype=np.uint8,
                              count=-1, offset=8).astype(np.int32)
        assert number_of_items == len(items), "Invalid items count"
        return items

    @staticmethod
    def parse_mnist_images_file(file_path):
        # return the labels
        with open(file_path, "rb") as f:
            data = f.read()
        # magic_number,number of items
        magic_number, number_of_items, rows, cols = np.frombuffer(
            data, dtype=">i", count=4, offset=0)

        # magic number is 2051
        assert magic_number == 2051, "Invalid images files"
        pixles = np.frombuffer(data, dtype=np.uint8,
                               count=-1, offset=16)
        images = pixles.reshape(number_of_items, cols, rows)
        return images


'''
This is dataset test 
'''
training_dataset = MNISTDataset('mnist/train-images.idx3-ubyte',
                                'mnist/train-labels.idx1-ubyte')
testing_dataset = MNISTDataset('mnist/t10k-images.idx3-ubyte',
                               'mnist/t10k-labels.idx1-ubyte')
print(f'num of training data {len(training_dataset)}')
print(f'num of testing data {len(testing_dataset)}')
img, label = training_dataset[0]
plt.imshow(img)
plt.title(f'training label is {label}')
plt.savefig(f'./pic/train_{label}.jpg', format='jpeg')
plt.show()
plt.close()

img, label = testing_dataset[0]
plt.imshow(img)
plt.title(f'testing label is {label}')
plt.savefig(f'./pic/test_{label}.jpg', format='jpeg')
plt.show()
plt.close()
