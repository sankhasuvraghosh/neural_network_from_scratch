import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
X,Y=spiral_data(samples=100,classes=3)
class Dropout:
    def __init__(self, rate):
        
        self.rate = 1 - rate

    def forward(self, inputs, training=True):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
 
        self.output = inputs * self.binary_mask

    def backward(self, d_values):
        self.dinputs = d_values * self.binary_mask
        
class adam_optimizer:
    def __init__(self, learning_rate=0.001, decay=0, 
                 epsilon=1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.epsilon               = epsilon
        self.beta1                 = beta1   
        self.beta2                 = beta2   

    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1 / (1.0 + self.decay * self.iterations))

    def update_parameters(self, layer):

       
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums   = np.zeros_like(layer.biases)
            layer.weight_cache     = np.zeros_like(layer.weights)
            layer.bias_cache       = np.zeros_like(layer.biases)

        
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (
                                 1 - self.beta1) * layer.dweights
        layer.bias_momentums   = self.beta1 * layer.bias_momentums + (
                                 1 - self.beta1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (
                                     1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected   = layer.bias_momentums   / (
                                     1 - self.beta1 ** (self.iterations + 1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (
                             1 - self.beta2) * layer.dweights**2
        layer.bias_cache   = self.beta2 * layer.bias_cache + (
                             1 - self.beta2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (
                                 1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected   = layer.bias_cache   / (
                                 1 - self.beta2 ** (self.iterations + 1))

        layer.weights += (-self.current_learning_rate * weight_momentums_corrected) / (
                          np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases  += (-self.current_learning_rate * bias_momentums_corrected)   / (
                          np.sqrt(bias_cache_corrected)  + self.epsilon)

    def post_update_parameters(self):
        self.iterations += 1
class rmsprop_optimizer:
    def __init__(self,learning_rate=0.001,decay=0,epsilon=1e-7,rho=0.9):
        self.decay=decay
        self.current_learning_rate=learning_rate
        self.learning_rate=learning_rate
        self.iterations=0
        self.epsilon=epsilon
        self.rho=rho
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate= self.learning_rate * (1 /(1.0 + (self.decay * self.iterations)))
    
    def update_parameters(self,layer):
        
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)
        layer.weight_cache =self.rho * layer.weight_cache +(1-self.rho) *layer.dweights**2
        layer.biases_cache = self.rho * layer.biases_cache +(1-self.rho) *layer.dbiases**2
        layer.weights += (-self.current_learning_rate * layer.dweights) / (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += (-self.current_learning_rate * layer.dbiases ) / (np.sqrt(layer.biases_cache)+self.epsilon)
        
    
    def post_update_parameters(self):
        self.iterations +=1
class adagrad_optimizer:
    def __init__(self,learning_rate=1,decay=0,epsilon=1e-7):
        self.decay=decay
        self.current_learning_rate=learning_rate
        self.learning_rate=learning_rate
        self.iterations=0
        self.epsilon=epsilon
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate /(1.0 + self.decay * self.iterations)
    
    def update_parameters(self,layer):
        
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.biases_cache=np.zeros_like(layer.biases)
        layer.weight_cache +=layer.dweights**2
        layer.biases_cache +=layer.dbiases**2
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.biases_cache)+self.epsilon)
        
    
    def post_update_parameters(self):
        self.iterations +=1
class sgd_optimizer:
    def __init__(self,learning_rate=1,decay=0,momentum=0):
        self.decay=decay
        self.current_learning_rate=learning_rate
        self.learning_rate=learning_rate
        self.iterations=0
        self.momentum=momentum
    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate /(1.0 + self.decay * self.iterations)
    
    def update_parameters(self,layer):
        if self.momentum:
            if not hasattr(layer,'weight_momentum'):
                layer.weight_momentum=np.zeros_like(layer.weights)
                layer.bias_momentum=np.zeros_like(layer.biases)
            weight_updates=self.momentum * layer.weight_momentum -( self.current_learning_rate * layer.dweights)
            layer.weight_momentum=weight_updates
            bias_updates=self.momentum * layer.bias_momentum -( self.current_learning_rate * layer.dbiases)
            layer.bias_momentum=bias_updates
        else :
            weight_updates=-self.current_learning_rate*layer.dweights
            bias_updates=-self.current_learning_rate*layer.dbiases
        layer.weights+=weight_updates
        layer.biases+=bias_updates
    
    def post_update_parameters(self):
        self.iterations +=1

class loss:
    def regularization_loss(self,layer):
        regularization_loss=0
        if layer.w_regularizer_l1 > 0:
            regularization_loss += layer.w_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        if layer.w_regularizer_l2 > 0:
            regularization_loss += layer.w_regularizer_l2 * np.sum(layer.weights**2)
        
        
        if layer.b_regularizer_l1 > 0:
            regularization_loss += layer.b_regularizer_l1 * np.sum(np.abs(layer.biases))
        
        
        if layer.b_regularizer_l2 > 0:
            regularization_loss += layer.b_regularizer_l2 * np.sum(layer.biases**2)
        
        return regularization_loss
    def calculate(self,output,Y):
        sample_loses=self.forward(output,Y)
        data_mean=np.mean(sample_loses)
        return data_mean
class layer_entropy_loss(loss):
    def backward(self,d_values,y_true):
        sample=len(d_values)
        labels=len(d_values[0])
        if len(y_true.shape) ==1:
            y_true=np.eye(labels)[y_true]
        self.dinputs=-y_true/d_values
        self.dinputs=self.dinputs/sample

    def forward(self,y_pred,y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,(1e-7),1-(1e-7))
        if y_true.ndim == 1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        elif y_true.ndim ==2:
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)

        nll=-np.log(correct_confidences)
        return nll
class softmax_activation:
    def forward(self,inputs):
        self.exp=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output=self.exp/np.sum(self.exp,axis=1,keepdims=True)
        return self.output
    def backward(self,d_values,y_true):
        sample=len(d_values)
        self.dinputs=d_values.copy()
        if len(y_true.shape) ==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs[range(sample),y_true]-=1
        self.dinputs=self.dinputs/sample
    
class relu:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.maximum(0,inputs)
        return self.output
    def backward(self,d_values):
        self.dinputs=d_values.copy()
        self.dinputs[self.inputs<=0]=0

class activation_loss_entropy:
    def __init__(self):
        self.activation=softmax_activation()
        self.loss=layer_entropy_loss()
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)
    def backward(self,d_values,y_true):
        sample=len(d_values)
        self.dinputs=d_values.copy()
        if len(y_true.shape) ==2:
            y_true=np.argmax(y_true,axis=1)
        self.dinputs[range(sample),y_true]-=1
        self.dinputs=self.dinputs/sample

class layer_dense:
    def __init__(self,n_inputs,n_neurons,w_regularizer_l1=0, b_regularizer_l1 =0,w_regularizer_l2=0,b_regularizer_l2=0):
        self.weights=0.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        self.w_regularizer_l1= w_regularizer_l1
        self.b_regularizer_l1 =b_regularizer_l1
        self.w_regularizer_l2=w_regularizer_l2
        self.b_regularizer_l2=b_regularizer_l2

    def forward(self,inputs):
        self.inputs=inputs
        self.output=np.dot(inputs,self.weights)+self.biases
    def backward(self,d_values):
        self.dweights=np.dot(self.inputs.T,d_values)
        self.dbiases=np.sum(d_values,axis=0,keepdims=True)

        if self.w_regularizer_l1 > 0:
            dl1=np.ones_like(self.weights)
            dl1[self.weights<0]=-1
            self.dweights+=self.w_regularizer_l1 * dl1
        if self.w_regularizer_l2 > 0:
            self.dweights+=2 * self.w_regularizer_l2*self.weights

        if self.b_regularizer_l1 > 0:
            dl1=np.ones_like(self.biases)
            dl1[self.biases<0]=-1
            self.dbiases+=self.b_regularizer_l1 * dl1
        if self.b_regularizer_l2 > 0:
            self.dbiases+=2 * self.b_regularizer_l2*self.biases
        self.dinputs=np.dot(d_values,self.weights.T)

dense1=layer_dense(2,64,w_regularizer_l2=5e-4,b_regularizer_l2=5e-4)
activation1=relu()
dropout1=Dropout(0.1)
opt=adam_optimizer(learning_rate=0.02,
                     beta1=0.9,
                     beta2=0.999,
                     decay=5e-4)
dense2=layer_dense(64,3,w_regularizer_l2=5e-4,
    b_regularizer_l2=5e-4)
activation2=softmax_activation()

loss_activation=activation_loss_entropy()
if Y.ndim == 2:
     Y = np.argmax(Y, axis=1)

for epoch in range (10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output,training=True)
    dense2.forward(dropout1.output)
   

    data_loss=loss_activation.forward(dense2.output,Y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1)+loss_activation.loss.regularization_loss(dense2)
    loss_total=data_loss + regularization_loss
    predictions = np.argmax(loss_activation.output, axis=1)
    
    accuracy = np.mean(predictions == Y)
    if not epoch % 100 :
        print(f'epoch:{epoch},'+
              f'acc:{accuracy:.3f},'+
              f'loss:{data_loss}'+
              f'learning rate{opt.current_learning_rate}')
    loss_activation.backward(loss_activation.output,Y)
    dense2.backward(loss_activation.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)
    opt.pre_update_parameters()
    opt.update_parameters(dense1)
    opt.update_parameters(dense2)
    opt.post_update_parameters()
print("____________________________________________________________________-")
x_test, y_test=spiral_data(samples=100,classes=3)
dense1.forward(x_test)
activation1.forward(dense1.output)
dropout1.forward(activation1.output,training=False)
dense2.forward(dropout1.output)

loss_2=loss_activation.forward(dense2.output,y_test)
print("____________________________________________________________________-")

if y_test.ndim == 2:
    y_test = np.argmax(y_test, axis=1)
predictions = np.argmax(loss_activation.output, axis=1)
accuracy    = np.mean(predictions == y_test)
print(f'acc:{accuracy:.3f}')