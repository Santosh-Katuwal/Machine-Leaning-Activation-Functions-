import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-10,10,100)

#sigmoidal activation function
fx_sig=1/(1+np.exp(-x))

#tanh activation functino
fx_tanh=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

#ReLU activation function
fx_relu=[]
for i in x:
    fx=max(0,i)
    fx_relu.append(fx)

#leaky relu
fx_leakyrelu=[]
for i in x:
    fx=max(0,i)
    fx_leakyrelu.append(fx)
    
#softmax
fx_softmax=np.exp(x)/np.sum(np.exp(x))

plt.subplot(3,2,1)
plt.plot(x,fx_sig)
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('fx')

plt.subplot(3,2,2)
plt.plot(x,fx_tanh)
plt.title('Tanh')
plt.xlabel('x')
plt.ylabel('fx')

plt.subplot(3,2,3)
plt.plot(x,fx_relu)
plt.title('ReLU')
plt.xlabel('x')
plt.ylabel('fx')

plt.subplot(3,2,4)
plt.plot(x,fx_leakyrelu)
plt.title('Leaky ReLU')
plt.xlabel('x')
plt.ylabel('fx')

plt.subplot(3,2,5)
plt.plot(x,fx_softmax)
plt.title('Softmax')
plt.xlabel('x')
plt.ylabel('fx')

plt.tight_layout()