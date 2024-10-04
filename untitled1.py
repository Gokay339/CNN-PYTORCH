import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image # resimleri preprosses yaparken kullanıcaz
import os


#%% 

def read_images(path,num_img):
    array = np.zeros([num_img,64*32])   # 64*32 resim boyutu
    i = 0
    for img in os.listdir(path):  # os.listdir(path): Belirtilen klasördeki dosya isimlerini döner
        img_path = path + "\\" + img  # Görüntünün tam dosya yolunu oluşturur
        img = Image.open(img_path, mode="r")
        data = np.asarray(img,dtype = "uint8") # Görüntüyü Numpy Dizisine Dönüştürme
        data = data.flatten() # data.flatten(): 2 boyutlu bir diziyi tek boyutlu bir diziye dönüştürür 
                              # (örneğin, 64x32'lik bir görüntü 2048 uzunluğunda bir diziye dönüşür).
        array[i,:] = data
        # Her bir görüntü için array'in i'inci satırına, düzleştirilmiş data dizisi yerleştirilir. 
        # Böylece her satır bir görüntüyü temsil eder.
        i += 1
    return array

# TRAİN ALANI
train_neg_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Train\neg"
num_train_neg_img = 43390
train_neg_array = read_images(train_neg_path, num_train_neg_img)

# TORCH VERİLERİNİ NUMPY VERİLERİNE ÇEVİREBİLİRİZ
# 1 boyutlu vektör 2 boyutlu matris , 3 4 5 6 10 boyutlular bunların genel ismi tensor
x_train_neg_tensor = torch.from_numpy(train_neg_array)
print("x : ",x_train_neg_tensor.size())

y_train_neg_tensor = torch.zeros(num_train_neg_img,dtype = torch.long)
print("y : ",y_train_neg_tensor.size())

# POZİTİFLER
train_pos_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Train\pos"
num_train_pos_img = 10208
train_pos_array = read_images(train_pos_path, num_train_pos_img)

x_train_pos_tensor = torch.from_numpy(train_pos_array)
print("x : ",x_train_pos_tensor.size())

y_train_pos_tensor = torch.ones(num_train_pos_img,dtype = torch.long)
print("y : ",y_train_pos_tensor.size())


# concat train  - pytorch kütüphanesinde concatenate = cat
# YLER GENELLİKLE LABEL  XLER GENELLİKLE RESİMLERİMİZ
x_train = torch.cat((x_train_neg_tensor,x_train_pos_tensor),0)
y_train = torch.cat((y_train_neg_tensor,y_train_pos_tensor),0)
print("x_train size : ",x_train.size())
print("y_train size : ",y_train.size())


# TEST ALANI

test_neg_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Test\neg"
num_test_neg_img = 22050
test_neg_array = read_images(test_neg_path, num_test_neg_img)

# TORCH VERİLERİNİ NUMPY VERİLERİNE ÇEVİREBİLİRİZ
# 1 boyutlu vektör 2 boyutlu matris , 3 4 5 6 10 boyutlular bunların genel ismi tensor
x_test_neg_tensor = torch.from_numpy(test_neg_array[:20855,:])
print("x : ",x_test_neg_tensor.size())

y_test_neg_tensor = torch.zeros(20855,dtype = torch.long)
print("y : ",y_test_neg_tensor.size())

# POZİTİFLER
test_pos_path = r"C:\Users\gokay\OneDrive\Masaüstü\DerinOgrenme_1\DerinOgrenme_Dersler\Derin Ogrenme 5.1\2) Deep Residual Network\LSIFIR\Classification\Test\pos"
num_test_pos_img = 5944
test_pos_array = read_images(test_pos_path, num_test_pos_img)

x_test_pos_tensor = torch.from_numpy(test_pos_array)
print("x : ",x_test_pos_tensor.size())

y_test_pos_tensor = torch.ones(num_test_pos_img,dtype = torch.long)
print("y : ",y_test_pos_tensor.size())


# concat train  - pytorch kütüphanesinde concatenate = cat
# YLER GENELLİKLE LABEL  XLER GENELLİKLE RESİMLERİMİZ
x_test = torch.cat((x_test_neg_tensor,x_test_pos_tensor),0)
y_test = torch.cat((y_test_neg_tensor,y_test_pos_tensor),0)
print("x_train size : ",x_test.size())
print("y_train size : ",y_test.size())


#%%

plt.imshow(x_train[45001,:].reshape(64,32),cmap="gray")


#%% C-N-N

num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.0001

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,10,5) # 1 = input channel , 10 = output channel filtreden çıkanlar , 5 = filtremiz 5x5
        self.pool1 = nn.MaxPool2d(2,2) #  Görüntü boyutunu küçültmek için kullanılıyor
        self.conv2 = nn.Conv2d(10,16,5)
        
        self.fc1 = nn.Linear(16*13*5,520)  # GİRDİ VE ÇIKTI BOYUTLARI
        self.fc2 = nn.Linear(520,130)
        self.fc3 = nn.Linear(130,num_classes)
        
    def forward(self,x):
        x = self.pool1(F.relu((self.conv1(x))))
        # İlk evrişim katmanı (conv1) uygulanıyor. Sonra ReLU aktivasyon fonksiyonu, ardından havuzlama (pool).
        x = self.pool1(F.relu(self.conv2(x)))
        
        x = x.view(-1,16*13*5)
        # Veriyi düzleştiriyoruz. Bu, görüntüden gelen bilgiyi tam bağlı katmanlara verebilmek için yapılıyor.
        x = F.relu(self.fc1(x)) # İlk tam bağlı katman
        x = F.relu(self.fc2(x)) # İkinci tam bağlı katman
        x = self.fc3(x) # Son katman, sınıfları tahmin eden katman
        return x
    
import torch.utils.data
train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train , batch_size=batch_size, shuffle=True)
test = torch.utils.data.TensorDataset(x_test,y_test)
testloader = torch.utils.data.DataLoader(test , batch_size=batch_size, shuffle=False)

net = Net()

#%%  LOSS AND OPTİMİZE ETME

criterion = nn.CrossEntropyLoss()
import torch.optim as optim
opti = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.8)

#%% train a network

start = time.time()
train_acc = []
test_acc = []
loss_list = []

use_gpu= False   # ekran kartınızı kullanıcaksanız True yapıcaksınız

for epoch in range(num_epochs):
    for i,data in enumerate(trainloader,0):
        
        inputs , labels = data
        inputs = inputs.view(batch_size,1,64,32) # view anlamı reshape , colorchannel 1 (renksiz)   64 ,32 boyut
        inputs = inputs.float()  #float çevirdik
        
        
        #gradient CNN MODELİNİN ÖĞRENMESİNİ SAĞLAYAN TÜREVLER
        opti.zero_grad() # başlangıçta sıfırlıyoruz
        
        #forward
        outputs = net(inputs)
        
        #loss
        loss = criterion(outputs,labels)

        # back 
        loss.backward()
        
        # update weights
        opti.step()
        
    #test
    correct = 0
    total =0
    with torch.no_grad():   # PyTorch'ta otomatik farklılaştırmayı devre dışı bırakmak için kullanılır.
                            # Artık back propogation kapanıyor yapamıyoruz
    
        for data in testloader:
            images , labels = data
            images = images.view(images.size(0),1,64,32)
            images=images.float()
            
            output =  net(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc1 = 100*correct/total
    print("Accuracy Test : ",acc1)
    test_acc.append(acc1)
    
    
    
    #train
    correct = 0
    total =0
    with torch.no_grad():   # PyTorch'ta otomatik farklılaştırmayı devre dışı bırakmak için kullanılır.
                            # Artık back propogation kapanıyor yapamıyoruz
    
        for data in trainloader:
            images , labels = data
            images = images.view(batch_size,1,64,32)
            images=images.float()
            
            output =  net(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc2 = 100*correct/total
    print("Accuracy Train : ",acc2)
    train_acc.append(acc2)
    
print("Train is done.")


end = time.time() # ne kadar süreceğine bakmamız için 
process_time = (end - start)/60
print("process_time : ",process_time)

#%% visualize
fig, ax1 = plt.subplots()

plt.plot(loss_list,label = "Loss",color = "black")

ax2 = ax1.twinx()

ax2.plot(np.array(test_acc)/100,label = "Test Acc",color="green")
ax2.plot(np.array(train_acc)/100,label = "Train Acc",color= "red")
ax1.legend()
ax2.legend()
ax1.set_xlabel('Epoch')
fig.tight_layout()
plt.title("Loss vs Test Accuracy")
plt.show()



















