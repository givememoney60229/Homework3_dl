import numpy as np
from abc import ABCMeta, abstractmethod
import os
import cv2
def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def load_data(filename):
  fh=open(filename,"r",encoding="utf-8")
  lines=fh.readlines()
  data=[]
  label=[]
  for line in lines:
      line=line.strip("\n")
      line=line.strip()
      words=line.split()
      imgs_path=words[0]
      labels=words[1]
      label.append(labels)
      data.append(imgs_path)
  return data,label 

def load_mydata(filename,width,height): 
  data,label=load_data(filename)
  xs = []
  
  for i in range(len(label)):
    image_dir="C:/Users/user/Desktop/"
    img_path=os.path.join(image_dir,data[i])
    image=cv2.imread(img_path)
    
    if image.ndim==2:
        image=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
    X=cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
    xs.append(X)
    

  Xtr = np.array(xs)
  Ytr = np.asarray(label,dtype=int)
  
  return Xtr, Ytr
class FullyConnectedLayer:
    def __init__(self, in_features, out_features):#Initializes the weights and biases with random values.
        self.weights = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features, 1)

    def forward(self, x):
        # Computes the forward pass of the activation function. Takes in an input x, applies the element-wise
        # maximum operation between the input multiplied by alpha and the input itself, then returns the result.
        self.input = x
        z = np.dot(self.weights, x) + self.bias
        return z

    def backward(self, grad_output, learning_rate=0.001):
        # Computes the backward pass of the activation function. Takes in the gradient of the loss with respect to the output of the activation function (grad_output), computes the gradient of the input with respect to the loss using the derivative of
        # the Leaky ReLU function, then returns the gradient of the input with respect to the loss.
       # print("-----------self.weights------------",np.shape(self.weights))
        #print("------------self.input-----------",np.shape(self.input))
        #print("------------grad_output-----------",np.shape(grad_output))
        #print("------------self.bias-----------",np.shape(grad_output))
        grad_weights = np.dot(np.reshape(grad_output,(self.weights.shape[0],-1)),np.reshape(self.input,(-1,self.weights.shape[1])))
        #grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(np.reshape(self.weights,(self.input.shape[0],-1)),np.reshape(grad_output,(-1,self.input.shape[1])))
        #print("---------grad_input--------------",np.shape(grad_input))
        #print("-----------grad_bias------------",np.shape(grad_bias))
        self.weights -= learning_rate * grad_weights
        #self.bias -= learning_rate * grad_bias.T
        self.grad=grad_weights
        return grad_input


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.input = x
        output = np.maximum(self.alpha*x, x)
        return output

    def backward(self, grad_output):
        
        grad_input = np.ones_like(self.input)
        grad_input[self.input < 0] = self.alpha
        return  grad_input
    

class AvgPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        n, c, h, w = x.shape
        padded_h = h + 2*self.padding
        padded_w = w + 2*self.padding

        # Pad the input with zeros
        x_padded = np.zeros((n, c, padded_h, padded_w))
        x_padded[:, :, self.padding:h+self.padding, self.padding:w+self.padding] = x

        # Compute the output size
        out_h = int((padded_h - self.kernel_size) / self.stride + 1)
        out_w = int((padded_w - self.kernel_size) / self.stride + 1)

        # Initialize the output
        output = np.zeros((n, c, out_h, out_w))

        # Compute the average over each kernel
        for i in range(out_h):
            for j in range(out_w):
                output[:, :, i, j] = np.mean(x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size], axis=(2, 3))

        return output

    def backward(self, grad_output):
        n, c, h, w = grad_output.shape
        padded_h = h + 2*self.padding
        padded_w = w + 2*self.padding

        # Initialize the gradient of the input
        grad_input = np.zeros((n, c, padded_h, padded_w))

        # Compute the stride for the input gradient
        stride_h = int(padded_h - self.kernel_size) // (h - 1)
        stride_w = int(padded_w - self.kernel_size) // (w - 1)

        # Compute the gradient for each kernel
        for i in range(h):
            for j in range(w):
                grad_input[:, :, i*stride_h:i*stride_h+self.kernel_size, j*stride_w:j*stride_w+self.kernel_size] += grad_output[:, :, i, j][:, :, np.newaxis, np.newaxis] / self.kernel_size / self.kernel_size

        # Remove the padding from the gradient of the input
        grad_input = grad_input[:, :, self.padding:h+self.padding, self.padding:w+self.padding]

        return grad_input    
class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, gradient):
        
        return gradient * self.output * (1 - self.output)
class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.mask = None

    def forward(self, x):
        n, c, h, w = x.shape
        padded_h = h + 2*self.padding
        padded_w = w + 2*self.padding

        # Pad the input with zeros
        x_padded = np.zeros((n, c, padded_h, padded_w))
        x_padded[:, :, self.padding:h+self.padding, self.padding:w+self.padding] = x

        # Compute the output size
        out_h = int((padded_h - self.kernel_size) / self.stride + 1)
        out_w = int((padded_w - self.kernel_size) / self.stride + 1)

        # Initialize the output and the mask
        output = np.zeros((n, c, out_h, out_w))
        self.mask = np.zeros((n, c, padded_h, padded_w))

        # Compute the maximum over each kernel
        for i in range(out_h):
            for j in range(out_w):
                window = x_padded[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                max_val = np.max(window, axis=(2, 3))
                output[:, :, i, j] = max_val
                mask = (window == max_val[:, :, np.newaxis, np.newaxis])
                self.mask[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] = mask

        return output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.mask)
        grad_output = grad_output[:, :, np.newaxis, np.newaxis]
        grad_output_flatten=grad_output.flatten()
        num_1=np.array(np.where(self.mask==1))
        DIFF=np.abs(num_1.shape[1]-grad_output_flatten.shape[0])
       
        bias_vector=np.random.rand(DIFF)
        grad_output_flatten=np.concatenate((grad_output_flatten,bias_vector),axis=0)
        
        # Compute the gradient for each kernel
        grad_input[np.where(self.mask==1)] = grad_output_flatten

        # Remove the padding from the gradient of the input
        #grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input
    
class BatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.batch_mean = None
        self.batch_var = None

    def forward(self, x, train=True):
        n, c, h, w = x.shape

        # Compute batch mean and variance
        if train:
            self.batch_mean = np.mean(x, axis=(0, 2, 3))
            self.batch_var = np.var(x, axis=(0, 2, 3))

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            self.batch_mean = self.running_mean
            self.batch_var = self.running_var

        # Normalize the input
        x_norm = (x - self.batch_mean.reshape((1, self.num_features, 1, 1))) / np.sqrt(self.batch_var.reshape((1, self.num_features, 1, 1)) + self.eps)

        # Scale and shift the input
        out = self.gamma.reshape((1, self.num_features, 1, 1)) * x_norm + self.beta.reshape((1, self.num_features, 1, 1))

        return out

    def backward(self, grad_output):
        n, c, h, w = grad_output.shape

        # Compute the gradient with respect to gamma and beta
        grad_gamma = np.sum(grad_output * self.batch_norm, axis=(0, 2, 3))
        grad_beta = np.sum(grad_output, axis=(0, 2, 3))

        # Compute the gradient with respect to the input
        grad_input_norm = grad_output * self.gamma.reshape((1, self.num_features, 1, 1))

        # Compute the gradient with respect to the batch normalization
        batch_std = np.sqrt(self.batch_var.reshape((1, self.num_features, 1, 1)) + self.eps)
        grad_batch_norm = grad_input_norm / batch_std

        # Compute the gradient with respect to the variance
        grad_var = np.sum(grad_input_norm * (self.batch_norm - self.batch_mean.reshape((1, self.num_features, 1, 1))) * -0.5 * np.power(batch_std, -3), axis=(0, 2, 3))

        # Compute the gradient with respect to the mean
        grad_mean = np.sum(grad_input_norm * -1 / batch_std, axis=(0, 2, 3)) + grad_var * np.mean(-2 * (self.batch_norm - self.batch_mean.reshape((1, self.num_features, 1, 1))), axis=(0, 2, 3))

        # Compute the gradient with respect to the input
        grad_input = grad_batch_norm + grad_var * 2 * (self.batch_norm - self.batch_mean.reshape((1, self.num_features, 1, 1))) / (n * h * w) + grad_mean / (n * h * w)

        return grad_input, grad_gamma, grad_beta

class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])

class Adam:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]
        self.t = 0

    def step(self, gradients):
        self.t += 1
        
        for i, gradient in enumerate(gradients):
           
            

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * gradient
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * np.power(gradient, 2)
            m_hat = self.m[i] / (1 - np.power(self.betas[0], self.t))
            v_hat = self.v[i] / (1 - np.power(self.betas[1], self.t))
            self.parameters[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)        

class SoftmaxLoss:
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """
        Compute the Softmax loss and its gradients.

        y_pred: numpy array of shape (batch_size, num_classes)
            Predicted class scores for each sample in the batch.
        y_true: numpy array of shape (batch_size,)
            True class labels for each sample in the batch (encoded as integers).

        Returns:
        loss: scalar
            The mean Softmax loss over the batch.
        gradients: numpy array of shape (batch_size, num_classes)
            The gradients of the Softmax loss w.r.t. y_pred.
        """
        # Compute softmax probabilities
        exp_scores = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Compute Softmax loss
        batch_size = y_pred.shape[0]
        log_probs = -np.log(probs[range(batch_size), y_true])
        loss = np.mean(log_probs)

        # Compute gradients
        gradients = probs
        gradients[range(batch_size), y_true] -= 1
        gradients /= batch_size

        return loss, gradients            


class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """
        Compute the cross-entropy loss and its gradients.

        y_pred: numpy array of shape (batch_size, num_classes)
            Predicted class scores for each sample in the batch.
        y_true: numpy array of shape (batch_size,)
            True class labels for each sample in the batch (encoded as integers).

        Returns:
        loss: scalar
            The mean cross-entropy loss over the batch.
        gradients: numpy array of shape (batch_size, num_classes)
            The gradients of the cross-entropy loss w.r.t. y_pred.
        """
        # Compute cross-entropy loss
        batch_size = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)  # Avoid log(0) errors
        log_probs = -np.log(y_pred[range(batch_size), y_true])
        loss = np.mean(log_probs)

        # Compute gradients
        gradients = y_pred
        gradients[range(batch_size), y_true] -= 1
        gradients /= batch_size

        return loss, gradients
    

class NLLLoss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """
        Compute the negative log-likelihood loss and its gradients.

        y_pred: numpy array of shape (batch_size, num_classes)
            Predicted class probabilities for each sample in the batch.
        y_true: numpy array of shape (batch_size,)
            True class labels for each sample in the batch (encoded as integers).

        Returns:
        loss: scalar
            The mean negative log-likelihood loss over the batch.
        gradients: numpy array of shape (batch_size, num_classes)
            The gradients of the negative log-likelihood loss w.r.t. y_pred.
        """
        # Compute negative log-likelihood loss
        batch_size = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)  # Avoid log(0) errors
        log_probs = -np.log(y_pred[range(batch_size), y_true])
        loss = np.mean(log_probs)

        # Compute gradients
        gradients = y_pred
        gradients[range(batch_size), y_true] -= 1
        gradients /= batch_size
        gradients *= -1

        return loss, gradients    
    
class MSELoss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true):
        """
        Compute the mean squared error loss and its gradients.

        y_pred: numpy array of shape (batch_size, num_classes)
            Predicted values for each sample in the batch.
        y_true: numpy array of shape (batch_size, num_classes)
            True values for each sample in the batch.

        Returns:
        loss: scalar
            The mean squared error loss over the batch.
        gradients: numpy array of shape (batch_size, num_classes)
            The gradients of the mean squared error loss w.r.t. y_pred.
        """
        # Compute mean squared error loss
        batch_size = y_pred.shape[0]
        errors = y_pred - y_true
        loss = np.mean(errors**2)

        # Compute gradients
        gradients = 2 * errors / batch_size

        return loss, gradients    
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.zeros((out_channels, 1))
        self.cache = None

    def forward(self, x):
        self.cache = x
        batch_size, in_channels, height, width = x.shape
        padded_height = height + 2*self.padding
        padded_width = width + 2*self.padding
        padded_x = np.zeros((batch_size, in_channels, padded_height, padded_width))
        padded_x[:, :, self.padding:padded_height-self.padding, self.padding:padded_width-self.padding] = x

        out_height = (padded_height - self.kernel_size)//self.stride + 1
        out_width = (padded_width - self.kernel_size)//self.stride + 1
        size="batch_size: {} \n self.out_channels: {} \n  out_height: {} \n out_width: {} \n ".format(batch_size, self.out_channels, out_height, out_width)
        #print(size)
        out = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                patch = padded_x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for k in range(self.out_channels):
                    out[:, k, i, j] = np.sum(patch*self.weights[k, :, :, :], axis=(1,2,3))
                out[:, :, i, j] += self.bias[:, 0]

        return out

    def backward(self, grad_out, learning_rate=0.3,I=False):
        x = self.cache
        self.grad=grad_out
        batch_size, in_channels, height, width = x.shape
        padded_height = height + 2*self.padding
        padded_width = width + 2*self.padding
        padded_x = np.zeros((batch_size, in_channels, padded_height, padded_width))
        padded_x[:, :, self.padding:padded_height-self.padding, self.padding:padded_width-self.padding] = x

        _, out_channels, out_height, out_width = grad_out.shape
        grad_x = np.zeros((batch_size, in_channels, height, width))
        grad_weights = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        grad_bias = np.zeros((self.out_channels, 1))
        

        for i in range(out_height):
            
            for j in range(out_width):
                patch = padded_x[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                for k in range(self.out_channels):
                    a=np.sum(patch * grad_out[:, k, i, j][:, np.newaxis, np.newaxis, np.newaxis], axis=0)
                    grad_weights[ k,:, :, :] += np.sum(a,axis=0)
                    
                grad_bias[:, 0] += np.sum(grad_out[:, :, i, j], axis=0)
                
                for n in range(batch_size):
                    for c in range(in_channels):
                        if I==True:
                            c=0
                        
                        grad_x[n, c, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += np.sum(np.sum(
                            grad_out[n, :, i, j][:, np.newaxis, np.newaxis, np.newaxis] * self.weights[:, c, :, :], axis=0), axis=0)
        self.gradient_weights=grad_weights
        self.gradient_bias=grad_bias
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        
        return grad_x  
    
class Softmax2:
    def __init__(self, dim):
        self.dim = dim
        
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        return e_x / np.sum(e_x, axis=self.dim, keepdims=True)
##use cource code ,it should be write by myself later    
class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def forward(self, X):
        #print("Softmax: _forward")
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def backward(self, dout):
        X, Y, Z = self.cache
        print("Z:",Z)
        print("size of Z:",np.shape(Z))
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(dout,dZ)
        dX = np.dot(dX,dY)
        return dX    

class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr
    
    def update(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= self.lr * grad

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

class LeNet5(Net):
    # LeNet5

    def __init__(self):
        self.conv1 = Conv2d(1, 6, 3)
        self.ReLU1 = LeakyReLU(0.02)
        self.pool1 = MaxPool2d(2,2)
        self.conv2 = Conv2d(6, 16, 3)
        self.ReLU2 = LeakyReLU(0.02)
        self.pool2 = MaxPool2d(2,2)
        self.FC1 = FullyConnectedLayer(400, 120)
        self.ReLU3 = LeakyReLU()
        self.FC2 = FullyConnectedLayer(120, 84)
        self.ReLU4 = LeakyReLU()
        self.FC3 = FullyConnectedLayer(84, 50)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X):
        h1 = self.conv1.forward(X)
        a1 = self.ReLU1.forward(h1)
        p1 = self.pool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.ReLU2.forward(h2)
        p2 = self.pool2.forward(a2)
        self.p2_shape = p2.shape
        fl = p2.reshape(X.shape[0],-1).transpose() # Flatten
        h3 = self.FC1.forward(fl)
        a3 = self.ReLU3.forward(h3)
        h4 = self.FC2.forward(a3)
        a5 = self.ReLU4.forward(h4)
        h5 = self.FC3.forward(a5)
        a5 = self.Softmax.forward(h5)


        return a5

    def backward(self, dout):
        #dout = self.Softmax.backward(dout)
        dout = self.FC3.backward(dout)
        dout = self.ReLU4.backward(dout)
        dout = self.FC2.backward(dout)
        dout = self.ReLU3.backward(dout)
        dout = self.FC1.backward(dout)
        dout = dout.reshape(self.p2_shape) # reshape
        dout = self.pool2.backward(dout)
        dout = self.ReLU2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.ReLU1.backward(dout)
        dout = self.conv1.backward(dout,I=True)

    def get_params(self):
        return [self.conv1.weights, self.conv1.bias, self.conv2.weights, self.conv2.bias, self.FC1.weights, self.FC2.weights, self.FC3.weights]
    
    def get_grad(self):
        return [self.conv1.gradient_weights, self.conv1.gradient_bias, self.conv2.gradient_weights, self.conv2.gradient_bias, self.FC1.grad, self.FC2.grad, self.FC3.grad]

    def set_params(self, params):
        [self.conv1.weights, self.conv1.bias, self.conv2.weights, self.conv2.bias, self.FC1.weights, self.FC1.bias, self.FC2.weights, self.FC2.bias, self.FC3.weights, self.FC3.bias] = params  


class LeNet5_enhanca(Net):
    # LeNet5

    def __init__(self):#3*32*32
        self.conv1 = Conv2d(3, 6, 3)#6*30*30
        self.Sigmoid1 = Sigmoid()
        self.pool1 = MaxPool2d(2,2)#6*15*15
        self.conv2 = Conv2d(6, 10, 4)#10*12*12
        self.Sigmoid2 = Sigmoid()
        self.pool2 = MaxPool2d(2,2)#10*6*6
        self.conv3 = Conv2d(10, 16, 3)#16*4*4
        self.Sigmoid3 = Sigmoid()
        self.pool3 = MaxPool2d(2,2)#16*2*2
        self.conv4 = Conv2d(16, 20, 1)#16*2*2
        self.Sigmoid4 = Sigmoid()
        self.FC1 = FullyConnectedLayer(80, 120)
        self.Sigmoid5 = Sigmoid()
        self.FC2 = FullyConnectedLayer(120, 84)
        self.Sigmoid6 = Sigmoid()
        self.FC3 = FullyConnectedLayer(84, 50)
        self.Softmax = Softmax()

        self.p2_shape = None

    def forward(self, X):
        h1 = self.conv1.forward(X)
        a1 = self.Sigmoid1.forward(h1)
        p1 = self.pool1.forward(a1)
        h2 = self.conv2.forward(p1)
        a2 = self.Sigmoid2.forward(h2)
        p2 = self.pool2.forward(a2)
        h3 = self.conv3.forward(p2)
        a3 = self.Sigmoid3.forward(h3)
        p3 = self.pool3.forward(a3)
        h4 = self.conv4.forward(p3)
        a4 = self.Sigmoid4.forward(h4)
        self.a4_shape = a4.shape
        fl = a4.reshape(X.shape[0],-1).transpose() # Flatten
        h3 = self.FC1.forward(fl)
        a3 = self.Sigmoid5.forward(h3)
        h4 = self.FC2.forward(a3)
        a5 = self.Sigmoid6.forward(h4)
        h5 = self.FC3.forward(a5)
        a5 = self.Softmax.forward(h5)


        return a5

    def backward(self, dout):
        #dout = self.Softmax.backward(dout)
        dout = self.FC3.backward(dout)
        dout = self.Sigmoid6.backward(dout)
        dout = self.FC2.backward(dout)
        dout = self.Sigmoid5.backward(dout)
        dout = self.FC1.backward(dout)
        dout = dout.reshape(self.a4_shape) # reshape
        dout = self.Sigmoid4.backward(dout)
        dout = self.conv4.backward(dout)
        dout = self.pool3.backward(dout)
        dout = self.Sigmoid3.backward(dout)
        dout = self.conv3.backward(dout)
        dout = self.pool2.backward(dout)
        dout = self.Sigmoid2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.pool1.backward(dout)
        dout = self.Sigmoid1.backward(dout)
        dout = self.conv1.backward(dout)

    def get_params(self):
        return [self.conv1.weights, self.conv1.bias, self.conv2.weights, self.conv2.bias, self.conv3.weights, self.conv3.bias, self.conv4.weights, self.conv4.bias, self.FC1.weights, self.FC2.weights, self.FC3.weights]
    
    def get_grad(self):
        return [self.conv1.gradient_weights, self.conv1.gradient_bias, self.conv2.gradient_weights, self.conv2.gradient_bias,self.conv3.gradient_weights, self.conv3.gradient_bias,self.conv4.gradient_weights, self.conv4.gradient_bias, self.FC1.grad, self.FC2.grad, self.FC3.grad]

    def set_params(self, params):
        [self.conv1.weights, self.conv1.bias, self.conv2.weights, self.conv2.bias,self.conv3.weights, self.conv3.bias, self.conv4.weights, self.conv4.bias, self.FC1.weights, self.FC1.bias, self.FC2.weights, self.FC2.bias, self.FC3.weights, self.FC3.bias] = params  
if __name__=="__main__":
    input_width=32
    input_height=32
    batch_size=500
    
    
    X_val, y_val =load_mydata("C:/Users/user/Desktop/val.txt",input_width,input_height)
    X_train, y_train =load_mydata("C:/Users/user/Desktop/train.txt",input_width,input_height)
    X_test, y_test =load_mydata("C:/Users/user/Desktop/test.txt",input_width,input_height)
    X_test=np.transpose(X_test,[0,3,1,2])
    X_train=np.transpose(X_train,[0,3,1,2])
    X_val=np.transpose(X_val,[0,3,1,2])
    toal_data_num=X_train.shape[0]
    # As a sanity check, we print out the size of the training and test data.
    losses = []
    model=LeNet5_enhanca()
    optim = Adam(model.get_params(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)
    criterion = NLLLoss()
    train_epoch=100
    for i in range(train_epoch):
        v=np.random.randint(toal_data_num, size=batch_size)
        data_train=X_train[v,:,:,:]
        label_train=y_train[v]
        Y_pred = model.forward(data_train)
        Y_pred=np.transpose(Y_pred,[1,0])
        loss, gradients = criterion.__call__(Y_pred, label_train)
        x_labels = "epoch: {}  loss: {}  ".format(i, loss)
        print(x_labels)
        model.backward(gradients)
        para_gradient=model.get_grad()
        optim.step(gradients=para_gradient)
        losses.append(loss)
       
    # TRAIN SET ACC
    Y_pred = model.forward(X_test)
    result = np.argmax(Y_pred, axis=0) - y_test
    result = list(result)
    print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))

    # TEST SET ACC
    Y_pred = model.forward(X_val)
    result = np.argmax(Y_pred, axis=0) - y_val
    result = list(result)
    print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_val.shape[0]) + ", acc=" + str(result.count(0)/X_val.shape[0]))
