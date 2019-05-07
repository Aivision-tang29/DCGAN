#!/usr/bin/env python
# coding: utf-8

# In[21]:


from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
batch_size=64
image_size=64
num_latent=100
num_generator_feature=32
num_discriminator_feature=32
num_iter=25
num_epoch=5
lr=0.0002
beta1=0.5
num_gpu=1
nc=3


# In[22]:


train_dataset=dataset.CIFAR10(root="./data",transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]))

dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[23]:


real_batch=next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Training images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=2,normalize=True).cpu(),(1,2,0)))


# In[24]:


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)


# In[25]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.main=nn.Sequential(
            nn.ConvTranspose2d(num_latent,num_generator_feature*8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(num_generator_feature*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_generator_feature*8,num_generator_feature*4,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_generator_feature*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_generator_feature*4,num_generator_feature*2,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_generator_feature*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_generator_feature*2,num_generator_feature,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_generator_feature),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_generator_feature,nc,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
        )
    def forward(self,x):
        return self.main(x)


# In[26]:


netG=Generator().to(device)
netG.apply(weight_init)
print(netG)


# In[27]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main=nn.Sequential(
            nn.Conv2d(nc,num_discriminator_feature,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_discriminator_feature,num_discriminator_feature*2,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_discriminator_feature*2,num_discriminator_feature*4,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_discriminator_feature*4,num_discriminator_feature*8,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_discriminator_feature*8,1,4,1,0,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.main(x)


# In[28]:


netD=Discriminator().to(device)
netD.apply(weight_init)
print(netD)


# In[29]:


criterion=nn.BCELoss()
fixed_noise=torch.randn(64,num_latent,1,1,device=device)

real_label=1
fake_label=0

optimizerD=optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
optimizerG=optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))


# In[ ]:


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epoch):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, num_latent, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epoch, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epoch-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


# In[ ]:


torch.save(netG.state_dict(),"netG.pth")
torch.save(netD.state_dict(),"netD.pth")


# In[ ]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[ ]:


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()


# In[ ]:




