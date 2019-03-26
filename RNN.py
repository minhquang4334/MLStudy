from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd, gluon, init
display.set_matplotlib_formats('svg')
embedding = 4 # Embedding dimension for autoregressive model
T = 1000 # Generate a total of 1000 points
time = nd.arange(0,T)
x = nd.sin(0.01 * time) + 0.2 * nd.random.normal(shape=(T))
plt.plot(time.asnumpy(), x.asnumpy())
plt.show()
