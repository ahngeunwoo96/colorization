import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from skimage.color import rgb2lab, rgb2gray, lab2rgb

use_gpu = torch.cuda.is_available()

print(use_gpu)

class GrayscaleImageFolder(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target
        
transform = transforms.Compose(
    [#transforms.Grayscale(num_output_channels=1),
     #transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip()])

#trainset = torchvision.datasets.ImageFolder(root = '/content/drive/MyDrive/dataset/train', transform = transform)
trainset = GrayscaleImageFolder('/content/drive/MyDrive/dataset/train', transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True, num_workers = 2)

validationset = GrayscaleImageFolder('/content/drive/MyDrive/dataset/test', transform)

validationloader = torch.utils.data.DataLoader(validationset, batch_size = 4, shuffle=False, num_workers = 2)

from google.colab import drive
drive.mount('/content/drive')

def visualize_image(grayscale_input, ab_input=None, show_image=False):
    '''Show or save image given grayscale (and ab color) inputs. Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib plot
    ab_input = ab_input.cpu()
    grayscale_input = grayscale_input.cpu()    
    if ab_input is None:
        grayscale_input = grayscale_input.squeeze().numpy() 
        if show_image: 
            plt.imshow(grayscale_input, cmap='gray')
            plt.show()
    else: 
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))  
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if show_image: 
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(grayscale_input, cmap='gray')
            axarr[1].imshow(color_image)
            plt.show()
 class Net(nn.Module):
    def __init__(self, down_sample = False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, 1, 1)  
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 2, 3, 1, 1)

    def forward(self, x):
        residual = x # [4, 1, 224, 224]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
      
        return x


net = Net()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# SGD : gradient descent, lr = learning rate 

for i, (gray_image, input_gt, target) in enumerate(trainloader):
   if i == 1:
     visualize_image(gray_image[i], input_gt[i].data, show_image = True)
     print(input_gt.shape)

net = net.cuda()

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, (grayimage, input_gt, target) in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        # inputs = inputs.cuda()
        # labels = labels.cuda()
        # input_gt = input.cuda()
        target = target.cuda()
        input_gray = grayimage.cuda()
        input_gt = input_gt.cuda()

        # forward + backward + optimize
        outputs = net(input_gray)
        loss = criterion(outputs, input_gt)

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

        #if i == 1:
        #  visualize_image(input_gray[i], outputs[i].data, show_image = True)

        #if epoch == 2 & i == 120000:
        #  visualize_image(input_gray[i], outputs[i].data, show_image = True)


PATH = './Colorization_net.pth'
torch.save(net.state_dict(), PATH)

print('Finished Training')

for i,(input_gray, input_gt, target) in enumerate(validationloader):
  print(i)
  net = Net()
  net.load_state_dict(torch.load(PATH))

  outputs = net(input_gray)

  if i == 0:
    visualize_image(input_gray[i], outputs[i].data, show_image=True)

  plt.clf()
  input_gt = input_gt.cpu()
  input_gray = input_gray.cpu()
  x = 0
  if i == 7:
    for j in range(2):
      color_image = torch.cat((input_gray[j], outputs[j].data), 0).numpy()
      color_image = color_image.transpose((1, 2, 0))  
      color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
      color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
      color_image = lab2rgb(color_image.astype(np.float64))
      #input_gray[i] = input_gray[i].squeeze().numpy()
      plt.imsave(arr=color_image, fname='{}{}'.format('/content/drive/MyDrive/output/','{}.png'.format(i * 4 + j)))
      x += 1
  else:
    for j in range(4):
      color_image = torch.cat((input_gray[j], outputs[j].data), 0).numpy()
      color_image = color_image.transpose((1, 2, 0))  
      color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
      color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
      color_image = lab2rgb(color_image.astype(np.float64))
      #input_gray[i] = input_gray[i].squeeze().numpy()
      plt.imsave(arr=color_image, fname='{}{}'.format('/content/drive/MyDrive/output/','{}.png'.format(i * 4 + j)))
      x += 1

  
     break
