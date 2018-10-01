## Walk through of metrics
Just to walk through anyone who maybe reading this repo. Inaccuracy is measured by the number of incorrect predicts over
the total number of predictions. Log loss (aka cross entropy) is the negative log of the probability of selecting 
the correct label. As the probability of the selecting the correct label approaches 1 then the -log_loss will approach 0
. Conversely, as the probability of selecting the correct label approaches 0, the -log_loss will approach infinity. 
Finally, the negative probability is the probability that the incorrect label is selected given an image of a digit. 
Ideally all of these metrics will be as close to 0 as possible.

## Equipment
All models were trained using google colab's GPU servers. At the moment of writing this, it consisted of 2 cpus, 12G of 
ram and a 12G K80 GPU.

## Models

### Basic Model
The base model that the other models inherited from flattened the input images it received, passed the flattened image
to a fully connected layer of 256 units to a relu activation function. The output of the relu layer is then passed to a
dropout layer with a 80% keep probability which is then passed a final fully connected layer of 10 units. The basic 
model uses this embedding/logits and determines the cross entropy loss. The cross entropy loss is what the models are 
optimized for. 

### 'Mixture of Softmaxes'
The 'mixture of softmaxes' (MoS) model is created by taking the flattened image and projecting it into _n_ components. 
This projection is done by passing the flattened image through a fully connected layer with a tanh activation unit and 
an output of (batch, 28 * 28 * _n_) == (batch, 784 * _n_). The resulting projection is split into the _n_ components 
and passed through the basic model as described above. The output of which is _n_ number of logits which are of shape 
(batch, number_of_classes) for the respective projected image. Thus the final logits are of shape 
(batch, number_of_classes, _n_). Since the output logits of the projected images have been through a non-linear unit we
can combine them by a weighted average at this stage. 

The prior or the weighting of each component is determined by taking the original input (the flattened image) and 
passing it through a fully connected layer with an output of _n_ units and passed through a softmax function. The 
weighting for each component is then multiplied with it's respective component to return a weighted average of the 
logits for the respective batch. From this point, the cross entropy loss can be determined the same as the basic model
above.

Although, the number of components can be arbitrarily high, we selected 3 components for the 'MoS' model. Also, it 
 should be noted that the original paper doesn't explicitly state it, the component matrix is implemented as a fully 
 connected layer with tanh activation rather than just a weights matrix with tanh activation. 

### Convolutional Neural Network
The convolutional neural network goes through two rounds of convolution and subsequent pooling steps. After the first 
conv + pooling layer we have a layer of (batch, 14, 14, 32). After the second conv and max pooling layer, we have an 
output of (batch, 7, 7, 64). This final output layer is then fed into the basic model same as the MoS model.

## Results
From the plots of the inaccuracy, log loss and negative probability, it can be seen that the order of models from worst
to best performing is the basic model, 'mixture of softmaxes' and finally the convolutional neural network. The same 
order is observed in the variability when evaluating the test data and for the amount of time that it took to train each
model for the same number of passes through the data.

## Discussion
The most interesting part of the results above, is the fact that the 'Mixture of softmaxes' model has a positive effect 
compared to the based line on classifying handwritten digits in the MNIST dataset. The original purpose of the mixture 
of softmaxes model intended to be for language modeling. More specifically, to increase the expressiveness of a model 
with a large potential vocabulary without increasing the embedding dimension size which could lead to overfitting.

Although technically, the 'MoS' model constructed is in fact not a mixture of softmaxes, as long as it is a mixture of 
logits after a non-linear unit then it is effective. If there is no non-linear units that acted on the individual logits
 then taking the weighted average of the components using the _prior_ (or component weight) will result in a mixture of 
 contexts which was shown to have no increase in rank or effectiveness. 
 
 The MoS model is nice in the sense that there are not a lot of extra parameters to learn. The only parameters needed 
 to be learned is the component (projection) matrix and the component weighting matrix. We can see the increase in 
 the number of training parameters increases the training time over the baseline model, but the 'MoS' model has a 
 shorter training time compared to the CNN model. The 'MoS' model adds _n(d<sup>2</sup>+2d+1)_ more parameters to be learned, 
 where _n_ is the number of components and _d_ is the embedding dimension size. The number of parameters to learn for 
 of the models is 203530, 2052205 and 857738. Although the CNN model has significantly less parameters to learn 
 compared to the MoS model, it takes 20% longer to train. 

## References
1. Yang, Zhilin, et al. "Breaking the softmax bottleneck: A high-rank RNN language model." arXiv preprint 
arXiv:1711.03953 (2017).

2. https://github.com/zihangdai/mos

