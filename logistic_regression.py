from utilities import *

train_x, train_y, test_x, test_y = load_data()

m = train_x.shape[1]
n0 = train_x.shape[0]

display(train_x, 75)
