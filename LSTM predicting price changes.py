import tensorflow as tf
import numpy as np
tf.reset_default_graph()

#define class Stock to pass parameters conveniently
class Stock():
    def _init_(self):
        # define the parameters for model
        self.training_epochs = 100
        self.batch_size = 1500
        self.learning_rate = 0.00003 
        self.loss_para = 0.0015
        
        self.hidden_layers = 32  
        self.out_layers = 1  
        #path for data input 
        self.xtrain_paths = "/Users/luffy/Desktop/DL/data/hushen30022/data_experiment/train/train_pct_chg.txt" 
        self.xtest_paths = "/Users/luffy/Desktop/DL/data/hushen30022/data_experiment/test/test_pct_chg.txt" 
        self.ytrain_path = "/Users/luffy/Desktop/DL/data/hushen30022/data_experiment/train/y_train.txt"
        self.ytest_path = "/Users/luffy/Desktop/DL/data/hushen30022/data_experiment/test/y_test.txt"
    
    #Input data
    def xload(self,xpath):
        file = open(xpath, 'r',encoding='utf-16')
        x = np.array(
        [i for i in [row.replace('\t', ' ').strip().split(' ') for row in file]],dtype=np.float32)
        file.close()
        return x 

    def yload(self,ypath):
        file = open(ypath, 'r',encoding='utf-16')
        y = np.array(
        [i for i in [row.replace('\t', ' ').strip().split(' ') for row in file]],dtype=np.float32)
        file.close()
        return y 
      

    def LSTM_Network(self,X): #Construct LSTM model
        self.weight_hidden=tf.Variable(tf.random_normal([self.input_num, self.hidden_layers]))
        self.weight_output=tf.Variable(tf.random_normal([self.hidden_layers, self.out_layers]))
        
        self.biases_hidden = tf.Variable(tf.random_normal([self.hidden_layers], mean=1.0))
        self.biases_output = tf.Variable(tf.random_normal([self.out_layers]))
        # Define two LSTM cells
        X = tf.nn.relu(tf.matmul(X, self.weight_hidden) + self.biases_hidden)
        X = tf.split(X, 1, 0)
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_layers, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.hidden_layers, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, X, dtype=tf.float32)
        return tf.matmul(outputs[-1], self.weight_output) + self.biases_output

    def train(self):
        self._init_()
        #input data
        self.xtrain = self.xload(self.xtrain_paths)
        self.xtest = self.xload(self.xtest_paths)
        self.ytrain = self.yload(self.ytrain_path)
        self.ytest = self.yload(self.ytest_path)
        self.train_num = len(self.xtrain)    
        self.input_num = len(self.xtrain[0]) 
        
        X = tf.placeholder(tf.float32, [None, self.input_num])
        Y = tf.placeholder(tf.float32, [None, self.out_layers])
        y_ = self.LSTM_Network(X)
        
        #Using regularization to avoid over-fitting
        L = self.loss_para * sum(tf.nn.l2_loss(i) for i in tf.trainable_variables())
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_)) + L 
        
        #optimizer,evaluation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        sess = tf.InteractiveSession()
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # train
        index=[i for i in range(len(self.ytrain))]
        print("Training begins")
        for i in range(self.training_epochs):
            #np.random.shuffle(index)
            for begin, end in zip(range(0, self.train_num, self.batch_size),
                                  range(self.batch_size, self.train_num + 1, self.batch_size)):
                temp=index[begin:end]
                sess.run(optimizer, feed_dict={X: self.xtrain[temp],Y: self.ytrain[temp]})
            print("train epoch: {},".format(i))
                

        #the train result 
        y_pred, loss_result = sess.run(
        [y_,  cost],feed_dict={X: self.xtest,Y: self.ytest})
        print("---end---")
        return y_pred,self.ytest
    


if __name__ == "__main__":
    stock=Stock()
    y_pred,y_truedata=stock.train()

    



