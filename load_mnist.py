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
# training_dataset = MNISTDataset('mnist/train-images.idx3-ubyte',
#                                 'mnist/train-labels.idx1-ubyte')
# testing_dataset = MNISTDataset('mnist/t10k-images.idx3-ubyte',
#                                'mnist/t10k-labels.idx1-ubyte')
# print(f'num of training data {len(training_dataset)}')
# print(f'num of testing data {len(testing_dataset)}')
# img, label = training_dataset[0]
# plt.imshow(img)
# plt.title(f'training label is {label}')
# plt.savefig(f'./pic/train_{label}.jpg', format='jpeg')
# plt.show()
# plt.close()

# img, label = testing_dataset[0]
# plt.imshow(img)
# plt.title(f'testing label is {label}')
# plt.savefig(f'./pic/test_{label}.jpg', format='jpeg')
# plt.show()
# plt.close()


class DataLoader:
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.index = np.arange(len(self.dataset))
        np.random.shuffle(self.index)
        self.cursor = 0
        return self

    def __next__(self):
        begin = self.cursor
        end = self.cursor+self.batch_size
        if end > len(self.dataset):
            raise StopIteration
        self.cursor = end
        batched_data = []
        for index in self.index[begin:end]:
            item = self.dataset[index]
            batched_data.append(item)
        return [np.stack(item, axis=0) for item in list(zip(*batched_data))]


'''
This is dataloader test 
'''
training_dataset = MNISTDataset('mnist/train-images.idx3-ubyte',
                                'mnist/train-labels.idx1-ubyte')
training_dataloader = DataLoader(dataset=training_dataset, batch_size=3)

testing_dataset = MNISTDataset('mnist/t10k-images.idx3-ubyte',
                               'mnist/t10k-labels.idx1-ubyte')
testing_dataloader = DataLoader(dataset=testing_dataset, batch_size=3)

for images, labels in training_dataloader:
    print(images.shape)
    print(labels.shape)
    plt.title(f"train label:{labels[0]}")
    plt.imshow(images[0])
    plt.show()
    break


for images, labels in testing_dataloader:
    print(images.shape)
    print(labels.shape)
    plt.title(f"test label:{labels[0]}")
    plt.imshow(images[0])
    plt.show()
    break
