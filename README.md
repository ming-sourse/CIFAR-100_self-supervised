# CIFAR-100_self-supervised
共四个代码，无需自己下载数据集，torch自带，首次使用时要联网，torch会将数据下载到对应的位置。  
pretrain_simclr.py是在STL-10数据集上进行全规模和部分规模预训练ResNet-18的代码。  
linear_classification.py是使用预训练好的ResNet-18用Linear Classification Protocol在CIFAR-100数据集上对其性能进行评测，注意将模型权重和代码放在统一路径下。  
supervised_pretrain.py是使用在ImageNet数据集上进行监督预训练的ResNet-18，并用Linear Classification Protocol对其性能进行评测。  
cifar100_supervised_train.py是在CIFAR-100数据集上从零开始监督训练ResNet-18得到的模型。
