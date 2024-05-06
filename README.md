# Replace MLP with Kolmogorov-Arnold Network and conduct experiments on CIFAR 10
A simple code to replace linear layers with KAN in CNN models. Experimental results on CIFAR10 show that the performance between CNNs with MLP and CNNs with KAN are similar.

The implentation of KAN are from: https://github.com/Blealtan/efficient-kan

| Model | Accuracy | 
| --- | --- | 
| VGG16+MLP(from scratch) | 0.8144 |  
| VGG16+KAN(from scratch)  |0.8177 |  
| VGG16+MLP(Pretrained)| 0.8897 |  
| VGG16+KAN(Pretrained)  | 0.8852 |  
