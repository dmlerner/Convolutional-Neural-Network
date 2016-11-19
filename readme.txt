A convolutional neural network to recognize MNIST handwritten digit dataset.
Compile with g++ -std=c++11 -fopenmp -O3 -o neural.out
Should converge to about 82% validation set error in ten minutes (97% test set)
Architecture is:
  28x28 inputs
  Convolve with 10x10 filter
  Add bias
  Maxpool 2x2 regions
  Fully connected layer down to 10 values
  Softmax
  Prediction is argmax of last layer
Uses mutual entropy loss function
Observed to not need weight decay
