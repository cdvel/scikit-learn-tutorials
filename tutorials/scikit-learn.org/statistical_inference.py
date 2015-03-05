print 'Loading digits dataset...'
from sklearn import datasets
digits = datasets.load_digits()
print 'dimensions of the digits dataset ', digits.images.shape
import pylab as pl
print 'printing with pylab...'
print pl.imshow(digits.images[-1], cmap=pl.cm.gray_r)
print '*'*15
print 'preprocessing dataset (2D array)'
newdata = digits.images.reshape(digits.images.shape[0], -1)
print 'dimensions after resizing', newdata.shape
print '*'*15