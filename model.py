import tensorflow as tf

class SobolevNetwork:
    def __init__(self, input_dim, num_hidden):
        self.input_dim = input_dim 
        self.num_hidden = num_hidden
        self.A1 = tf.Variable(tf.random.normal(
            [self.input_dim, self.num_hidden], mean=1.0, stddev=0.1))
        self.b1 = tf.Variable(tf.random.normal(
            [self.input_dim, self.num_hidden], mean=0.0, stddev=0.1))
        self.A2 = tf.Variable(tf.random.normal(
            [self.num_hidden, 1], mean=0.5, stddev=0.5))
        self.b2 = tf.Variable(tf.random.normal(
            [1, 1], mean=0.0, stddev=0.1))
       
    def forward(self, X):
        out = X
        out = tf.nn.softsign(tf.matmul(out, self.A1) + self.b1)
        out = tf.matmul(out, self.A2) + self.b2
        return out
    
    def grad_SS(self, X):
        out = X
        out = tf.matmul(out, self.A1) + self.b1
        out = 1/((1+tf.abs(out))**2)
        A = tf.transpose(self.A1)*(self.A2)
        out = tf.matmul(out, A)
        return out

