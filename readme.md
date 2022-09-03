# PyTorch Model Parameters Summary

#### Install using pip
```
pip install pytorchsummary
```
## WORKS ON CNNs and MLPs

NOTE: `summary()` functions returns a Tuple for (Total_trainable_params, Total_parameters, Total_non_trainable_params)

## Example 1

```python
from torch import nn
from pytorchsummary import summary

class CNNET(nn.Module):
    def __init__(self):
        super(CNNET,self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3,16,5), # 28-5+1
            nn.ReLU(), #24
            nn.MaxPool2d(2,2), # 12

            nn.Conv2d(16,32,3), # 12+1-3
            nn.ReLU(), # 10
            nn.MaxPool2d(2,2), # 5
            

            nn.Conv2d(32,64,5), # 11-3+1
            nn.ReLU(),

            nn.Conv2d(64,10,1)   
        )
    
    def forward(self,x):
        x = self.layer(x)
        return x

m = CNNET()
summary((3,128,128),m) 
```

### Output
```               Layer	Output Shape        	    Kernal Shape    	#params             	#(weights + bias)   	requires_grad
------------------------------------------------------------------------------------------------------------------------------------------------------
            Conv2d-1	[1, 16, 124, 124]   	   [16, 3, 5, 5]    	1216                	(1200 + 16)         	True True 
              ReLU-2	[1, 16, 124, 124]   	                    	                    	                    	          
         MaxPool2d-3	[1, 16, 62, 62]     	                    	                    	                    	          
            Conv2d-4	[1, 32, 60, 60]     	   [32, 16, 3, 3]   	4640                	(4608 + 32)         	True True 
              ReLU-5	[1, 32, 60, 60]     	                    	                    	                    	          
         MaxPool2d-6	[1, 32, 30, 30]     	                    	                    	                    	          
            Conv2d-7	[1, 64, 26, 26]     	   [64, 32, 5, 5]   	51264               	(51200 + 64)        	True True 
              ReLU-8	[1, 64, 26, 26]     	                    	                    	                    	          
            Conv2d-9	[1, 10, 26, 26]     	   [10, 64, 1, 1]   	650                 	(640 + 10)          	True True 
______________________________________________________________________________________________________________________________________________________

Total parameters 57,770
Total Non-Trainable parameters 0
Total Trainable parameters 57,770
(57770, 57770, 0)
```

```python
for i,j in enumerate(m.parameters()):
    if i==2:
        break
    j.requires_grad=False 
summary((3,128,128),m,border=True) 

```
```
              Layer	Output Shape        	    Kernal Shape    	#params             	#(weights + bias)   	requires_grad
------------------------------------------------------------------------------------------------------------------------------------------------------
            Conv2d-1	[1, 16, 124, 124]   	   [16, 3, 5, 5]    	1216                	(1200 + 16)         	False False
______________________________________________________________________________________________________________________________________________________
              ReLU-2	[1, 16, 124, 124]   	                    	                    	                    	          
______________________________________________________________________________________________________________________________________________________
         MaxPool2d-3	[1, 16, 62, 62]     	                    	                    	                    	          
______________________________________________________________________________________________________________________________________________________
            Conv2d-4	[1, 32, 60, 60]     	   [32, 16, 3, 3]   	4640                	(4608 + 32)         	True True 
______________________________________________________________________________________________________________________________________________________
              ReLU-5	[1, 32, 60, 60]     	                    	                    	                    	          
______________________________________________________________________________________________________________________________________________________
         MaxPool2d-6	[1, 32, 30, 30]     	                    	                    	                    	          
______________________________________________________________________________________________________________________________________________________
            Conv2d-7	[1, 64, 26, 26]     	   [64, 32, 5, 5]   	51264               	(51200 + 64)        	True True 
______________________________________________________________________________________________________________________________________________________
              ReLU-8	[1, 64, 26, 26]     	                    	                    	                    	          
______________________________________________________________________________________________________________________________________________________
            Conv2d-9	[1, 10, 26, 26]     	   [10, 64, 1, 1]   	650                 	(640 + 10)          	True True 
______________________________________________________________________________________________________________________________________________________
______________________________________________________________________________________________________________________________________________________

Total parameters 57,770
Total Non-Trainable parameters 1,216
Total Trainable parameters 56,554
(56554, 57770, 1216)
```



## Example 2
```python
from torchvision import models
from pytorchsummary import summary

m = models.alexnet(False)
summary((3,224,224),m)
# this function returns the total number of 
# parameters (int) in a model
```
### ouput
```
               Layer	Output Shape        	    Kernal Shape    	#params             	#(weights + bias)   	requires_grad
------------------------------------------------------------------------------------------------------------------------------------------------------
            Conv2d-1	[1, 64, 55, 55]     	  [64, 3, 11, 11]   	23296               	(23232 + 64)        	True True 
              ReLU-2	[1, 64, 55, 55]     	                    	                    	                    	          
         MaxPool2d-3	[1, 64, 27, 27]     	                    	                    	                    	          
            Conv2d-4	[1, 192, 27, 27]    	  [192, 64, 5, 5]   	307392              	(307200 + 192)      	True True 
              ReLU-5	[1, 192, 27, 27]    	                    	                    	                    	          
         MaxPool2d-6	[1, 192, 13, 13]    	                    	                    	                    	          
            Conv2d-7	[1, 384, 13, 13]    	  [384, 192, 3, 3]  	663936              	(663552 + 384)      	True True 
              ReLU-8	[1, 384, 13, 13]    	                    	                    	                    	          
            Conv2d-9	[1, 256, 13, 13]    	  [256, 384, 3, 3]  	884992              	(884736 + 256)      	True True 
             ReLU-10	[1, 256, 13, 13]    	                    	                    	                    	          
           Conv2d-11	[1, 256, 13, 13]    	  [256, 256, 3, 3]  	590080              	(589824 + 256)      	True True 
             ReLU-12	[1, 256, 13, 13]    	                    	                    	                    	          
        MaxPool2d-13	[1, 256, 6, 6]      	                    	                    	                    	          
AdaptiveAvgPool2d-14	[1, 256, 6, 6]      	                    	                    	                    	          
          Dropout-15	[1, 9216]           	                    	                    	                    	          
           Linear-16	[1, 4096]           	    [4096, 9216]    	37752832            	(37748736 + 4096)   	True True 
             ReLU-17	[1, 4096]           	                    	                    	                    	          
          Dropout-18	[1, 4096]           	                    	                    	                    	          
           Linear-19	[1, 4096]           	    [4096, 4096]    	16781312            	(16777216 + 4096)   	True True 
             ReLU-20	[1, 4096]           	                    	                    	                    	          
           Linear-21	[1, 1000]           	    [1000, 4096]    	4097000             	(4096000 + 1000)    	True True 
______________________________________________________________________________________________________________________________________________________

Total parameters 61,100,840
Total Non-Trainable parameters 0
Total Trainable parameters 61,100,840
(61100840, 61100840, 0)
```

### Calculating the number of specific layer, or layer frequencies
```python
from pytorchsummary import get_num_layers
print(get_num_layers(m)) # alexnet model 
```
Output:
```
{'Conv2d': 5,
 'ReLU': 7,
 'MaxPool2d': 3,
 'AdaptiveAvgPool2d': 1,
 'Dropout': 2,
 'Linear': 3}
 ```


# Example For MLPs

```python

from pytorchsummary import summary
from torch import nn

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.l = nn.Sequential(
        nn.Linear(18,16),
        nn.ReLU(),
        nn.Linear(16,8),
        nn.ReLU(),
        nn.Linear(8,4)
    )
  def forward(self,x):
    return self.l(x)
  
model = Net()
summary((18,),model) 

```
`summary()` function takes inputsize as a tuple so
**if len(input_size)==1**
you have to use `,` like this
`((input_size,))`

Otherwise it will throw an error

### output
```
               Layer	Output Shape        	    Kernal Shape    	#params             	#(weights + bias)   	requires_grad
------------------------------------------------------------------------------------------------------------------------------------------------------
            Linear-1	[1, 16]             	      [16, 18]      	304                 	(288 + 16)          	True True 
              ReLU-2	[1, 16]             	                    	                    	                    	          
            Linear-3	[1, 8]              	      [8, 16]       	136                 	(128 + 8)           	True True 
              ReLU-4	[1, 8]              	                    	                    	                    	          
            Linear-5	[1, 4]              	       [4, 8]       	36                  	(32 + 4)            	True True 
______________________________________________________________________________________________________________________________________________________

Total parameters 476
Total Non-Trainable parameters 0
Total Trainable parameters 476
(476, 476, 0)
```
