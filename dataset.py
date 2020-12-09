from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils import index2noun, to_var, build_graph
from dataset_utils import *


class DriveData(Dataset):
    '''This class charecterizes the key features of the dataset to be created
    For more details refer to the follwoing link
    https://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset'''

    __xs = []
    __ys = []
    __verb = []

    def __init__(self, image_ids, transform=None):

        self.transform = transform
        dataset_folder = 'Dataset/imsitu/of500_images_resized/'

        for key in image_ids.keys():
            self.__xs.append(dataset_folder + key)
            image_verb = image_ids[key]['verb']
            roles = verb2roles[image_verb]
            labels = []

            for an_idx in range(0, 3):
                temp = []
                for role in roles:
                    noun_id = image_ids[key]['frames'][an_idx][role]
                    temp.append(noun2index.get(
                        nouns[noun_id]['gloss'][0], noun2index['fake_noun']))
                temp = temp + [noun2index['fake_noun']] * \
                    (max_num_nodes - len(temp))
                labels = labels + [temp]

            self.__ys.append(labels)

            self.__verb.append(verb2index[image_ids[key]['verb']])

    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        verb = torch.from_numpy(np.asarray(self.__verb[index]).reshape([1, 1]))
        return img, label, verb

    def __len__(self):
        return len(self.__xs)


# In[14]:


class DriveData_test(Dataset):
    '''This class charecterizes the key features of the dataset to be created
    For more details refer to the follwoing link
    https://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset'''

    __xs = []
    __ys = []
    __verb = []

    def __init__(self, image_ids, transform=None):

        self.transform = transform
        dataset_folder = 'Dataset/imsitu/of500_images_resized/'

        for key in image_ids.keys():
            self.__xs.append(dataset_folder + key)
            image_verb = image_ids[key]['verb']
            roles = verb2roles[image_verb]

            labels = []

            for an_idx in range(0, 3):
                temp = []
                for role in roles:
                    noun_id = image_ids[key]['frames'][an_idx][role]
                    temp.append(noun2index.get(
                        nouns[noun_id]['gloss'][0], noun2index['fake_noun']))
                temp = temp + [noun2index['fake_noun']] * \
                    (max_num_nodes - len(temp))
                labels = labels + [temp]

            self.__ys.append(labels)

            self.__verb.append(verb2index[image_ids[key]['verb']])

    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        verb = torch.from_numpy(np.asarray(self.__verb[index]).reshape([1, 1]))
        return img, label, verb

    def __len__(self):
        return len(self.__xs)
