import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import data
from model import SobolevNetwork
tf.compat.v1.disable_eager_execution()

# set parameters
INPUT_DIM = 1
NUM_SAMPLES = 1000
NUM_HIDDEN = 2
NUM_EPOCHS = 1000

# initialize networks
X = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_der = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM])

model = SobolevNetwork(INPUT_DIM, NUM_HIDDEN)
y_p = model.forward(X)
dy = model.grad_SS(X)
A2 = model.A2

X_S = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_S = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM])
y_der_S = tf.compat.v1.placeholder(tf.float32, shape=[None, INPUT_DIM])

model_S = SobolevNetwork(INPUT_DIM, NUM_HIDDEN)
y_p_S = model_S.forward(X_S)
dy_S = model_S.grad_SS(X_S)
A2_S = model_S.A2

loss_N = tf.reduce_sum(input_tensor=tf.pow(y_p - y, 2))
loss_N_track = tf.reduce_sum(input_tensor=tf.pow(y_p_S - y_S, 2))

loss_S = tf.reduce_sum(tf.pow(y_p_S - y_S, 2)) + \
    tf.reduce_sum(tf.pow(dy_S - y_der_S, 2))
loss_S_track = tf.reduce_sum(tf.pow(y_p - y, 2)) + \
    tf.reduce_sum(tf.pow(dy - y_der, 2))
    
train_N = tf.compat.v1.train.AdamOptimizer(0.005).minimize(loss_N)
train_S = tf.compat.v1.train.AdamOptimizer(0.005).minimize(loss_S)

sess = tf.compat.v1.Session()
init = tf.compat.v1.global_variables_initializer()
sess.run(init)

plotA2 = []
plotL = []
plotL_S_track = []

plotA2_S = []
plotL_N_track = []
plotL_S = []


# TRAINING
for epoch_num, epoch in enumerate(range(NUM_EPOCHS)):
    # generate training data
    batch_samples = data.genTrainData_SS(num_samples=NUM_SAMPLES)
    X_train = [s[0] for s in batch_samples]
    y_train = [s[2] for s in batch_samples]     # learn softsign derivative
    dy_train = [s[3] for s in batch_samples]    # 2nd derivative
    
    # normal training
    train_dict = {X: X_train, y: y_train, y_der: dy_train}
    _, curr_loss, track_loss, y_preds = sess.run(
        [train_N, loss_N, loss_S_track, y_p], feed_dict=train_dict)
    plotA2.append(np.linalg.norm(sess.run(A2)))
    plotL.append(curr_loss)
    plotL_S_track.append(track_loss)
    print("Epoch: %d, Loss: %f" % (epoch_num+1, curr_loss))
    
    # Sobolev training  
    train_dict_S = {X_S: X_train, y_S: y_train, y_der_S: dy_train}
    _, track_loss, curr_loss, y_preds, dy_preds = sess.run(
        [train_S, loss_N_track, loss_S, y_p_S, dy_S], feed_dict=train_dict_S)
    plotA2_S.append(np.linalg.norm(sess.run(A2_S)))
    plotL_N_track.append(track_loss)
    plotL_S.append(curr_loss)
    print("Sobolev -- Epoch: %d, Loss: %f" % (epoch_num+1, curr_loss))
    

#%% PLOT
   
# Norm of A2 -- No Sobolev Training
plt.figure()
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotA2)
plt.title("Norm of A2 -- No Sobolev Training")
ax = plt.gca()
ax.ticklabel_format(useOffset=False)

# Loss -- No Sobolev Training
plt.figure()
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL)
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL_S_track)
plt.title("Loss -- No Sobolev Training")
ax = plt.gca()
ax.legend(['L2 Loss','Sobolev Loss'])
ax.ticklabel_format(useOffset=False)


# Target Function -- No Sobolev Training
xplot = np.linspace(-5, 5, num=101)
xplot = np.reshape(xplot, (xplot.shape[0],INPUT_DIM))
xplot = tf.cast(xplot, tf.float32)
yplot = model.forward(xplot)
yplot = yplot.eval(session=sess)
xplot = np.linspace(-5, 5, num=101)

plt.figure()
f = data.SSderiv()
plt.plot(xplot, f(xplot), 'blue', linewidth=5)
plt.plot(xplot, yplot, 'orange')
ax = plt.gca()
ax.legend(['Target','Network'])
plt.axis([-5, 5, -1, 1])
plt.title("")


# Norm of A2 -- Sobolev Training
plt.figure()
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotA2_S)
plt.title("Norm of A2 -- Sobolev Training")
ax = plt.gca()
ax.ticklabel_format(useOffset=False)

# Loss -- Sobolev Training
plt.figure()
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL_N_track)
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL_S)
plt.title("Loss -- Sobolev Training")
ax = plt.gca()
ax.legend(['L2 Loss','Sobolev Loss'])
ax.ticklabel_format(useOffset=False)


# Target Function -- Sobolev Training
xplot = np.linspace(-5, 5, num=101)
xplot = np.reshape(xplot, (xplot.shape[0],INPUT_DIM))
xplot = tf.cast(xplot, tf.float32)
yplot_S = model_S.forward(xplot)
yplot_S = yplot_S.eval(session=sess)
xplot = np.linspace(-5, 5, num=101)

plt.figure()
f = data.SSderiv()
plt.plot(xplot, f(xplot), 'blue', linewidth=5)
plt.plot(xplot, yplot_S, 'orange')
ax = plt.gca()
ax.legend(['Target','Network'])
plt.axis([-5, 5, -1, 1])
plt.title("Target Function -- Sobolev Training")


# Comparing L2 Losses
plt.figure()
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL)
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL_N_track)
plt.title("Comparing L2 Losses")
ax = plt.gca()
ax.legend(['L2 Loss, No Sobolev Training', 
           'L2 Loss with Sobolev Training'])
ax.ticklabel_format(useOffset=False)

# Comparing Sobolev Losses
plt.figure()
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL_S_track)
plt.plot(np.linspace(1, NUM_EPOCHS, num=NUM_EPOCHS), plotL_S)
plt.title("Comparing Sobolev Losses")
ax = plt.gca()
ax.legend(['Sobolev Loss, No Sobolev Training', 
           'Sobolev Loss with Sobolev Training'])
ax.ticklabel_format(useOffset=False)

