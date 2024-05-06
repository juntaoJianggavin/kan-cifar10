# Replace MLP with Kolmogorov-Arnold Network in CNN models and conduct experiments on CIFAR 10
A simple code to replace linear layers with KAN as the output layer in CNN models. Experimental results on CIFAR10 show that the performance of CNNs with MLP as the output layer and the performance of CNNs with KAN as the output layer are similar.

The implentation of KAN are from: https://github.com/Blealtan/efficient-kan

| Model | Accuracy | 
| --- | --- | 
| VGG16+MLP (from scratch) | 0.8144 |  
| VGG16+KAN (from scratch)  |0.8177 |  
| VGG16+MLP (Pretrained)| 0.8897 |  
| VGG16+KAN (Pretrained)  | 0.8852 |  
