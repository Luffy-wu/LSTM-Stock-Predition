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
        self.out_layers = 3  
        #path for data input 
        input_types= ["open.txt","high.txt","low.txt","close.txt",
                          "volume.txt","amt.txt","pct_chg.txt","turn.txt"]
        self.xtrain_paths = ["/Users/luffy/Desktop/DL/data/hushen3002/data_experiment/train/"
                             + "train_" + i for i in input_types]
        self.xtest_paths = ["/Users/luffy/Desktop/DL/data/hushen3002/data_experiment/test/"
                             + "test_" + i for i in input_types]
        self.ytrain_path = "/Users/luffy/Desktop/DL/data/hushen3002/data_experiment/train/y_train.txt"
        self.ytest_path = "/Users/luffy/Desktop/DL/data/hushen3002/data_experiment/test/y_test.txt"
    
    #Input data
    def xload(self,xpaths):
        X = []
        for path_ in xpaths:
            file = open(path_, 'r',encoding='utf-16')
            X.append([np.array(i, dtype=np.float32) for i in 
                       [row.replace('\t', ' ').strip().split(' ') for row in file]])
            file.close()
        return np.transpose(np.array(X), (1, 2, 0))

    def yload(self,ypath):
        file = open(ypath, 'r',encoding='utf-16')
        y = np.array(
        [i for i in [row.replace('\t', ' ').strip().split(' ') for row in file]],dtype=np.int32)
        file.close()
        return y - 1
    
    #Transform the label to one-hot values
    #eg: [2] --> [0, 0, 1]
    def one_hot(self,y):
        y = y.reshape(len(y))
        temp = int(np.max(y)) + 1
        return np.eye(temp)[np.array(y, dtype=np.int32)]  

    def RNN_Network(self,X): #Construct RNN model
        self.weight_hidden=tf.Variable(tf.random_normal([self.input_num, self.hidden_layers]))
        self.weight_output=tf.Variable(tf.random_normal([self.hidden_layers, self.out_layers]))
        
        self.biases_hidden = tf.Variable(tf.random_normal([self.hidden_layers], mean=1.0))
        self.biases_output = tf.Variable(tf.random_normal([self.out_layers]))
        tf.summary.histogram('weight_hidden', self.weight_hidden)
        tf.summary.histogram('weight_output', self.weight_output)
        tf.summary.histogram('biase_hidden', self.biases_hidden)
        tf.summary.histogram('biase_output', self.biases_output)
        
        # Reshape X for input,dimension_num*batch_size, n_input
        X = tf.transpose(X, [1, 0, 2])  
        X = tf.reshape(X, [-1, self.input_num])
        # Define two LSTM cells
        X = tf.nn.relu(tf.matmul(X, self.weight_hidden) + self.biases_hidden)
        X = tf.split(X, self.dimension_num, 0)
        rnn_cell_1 = tf.contrib.rnn.BasicRNNCell(config.n_hidden)
        rnn_cell_2 = tf.contrib.rnn.BasicRNNCell(config.n_hidden)
        rnn_cells = tf.contrib.rnn.MultiRNNCell([rnn_cell_1, rnn_cell_2], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(rnn_cells, _X, dtype=tf.float32)

        return tf.matmul(outputs[-1], self.weight_output) + self.biases_output

    def train(self):
        self._init_()
        #input data
        self.xtrain = self.xload(self.xtrain_paths)
        self.xtest = self.xload(self.xtest_paths)
        self.ytrain = self.one_hot(self.yload(self.ytrain_path))
        self.ytest = self.one_hot(self.yload(self.ytest_path))
        self.train_num = len(self.xtrain)  
        self.dimension_num = len(self.xtrain[0])  
        self.input_num = len(self.xtrain[0][0]) 
        
        X = tf.placeholder(tf.float32, [None, self.dimension_num, self.input_num])
        Y = tf.placeholder(tf.float32, [None, self.out_layers])
        y_ = self.RNN_Network(X)
        
        #Using regularization to avoid over-fitting
        L = self.loss_para * sum(tf.nn.l2_loss(i) for i in tf.trainable_variables())
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_)) + L 
        tf.summary.scalar('loss', cost)
        #optimizer,evaluation
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()
        sess = tf.InteractiveSession()
        log_dir='/Users/luffy/Desktop/DL/tensorboard/'
        train_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir+'test/')
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # train
        optimal_accuracy = 0.0
        index=[i for i in range(len(self.ytrain))]
        print("Training begins")
        j=0
        for i in range(self.training_epochs):
            np.random.shuffle(index)
            for begin, end in zip(range(0, self.train_num, self.batch_size),
                                  range(self.batch_size, self.train_num + 1, self.batch_size)):
                temp=index[begin:end]
                j += 1
                _,train_summary =sess.run([optimizer,merged], feed_dict={X: self.xtrain[temp],Y: self.ytrain[temp]})
                train_writer.add_summary(train_summary, j)
            #print the train result for each epoch
            y_pred, acc_result, loss_result,test_summary = sess.run(
            [y_, accuracy, cost,merged],feed_dict={X: self.xtest,Y: self.ytest})
            test_writer.add_summary(test_summary, i)
            print("train epoch: %d, test accuracy: %g, loss: %g"%(i,acc_result,loss_result))
            optimal_accuracy = max(optimal_accuracy, acc_result)
        train_writer.close()
        test_writer.close()
        print("final test accuracy: %g"%(acc_result))
        print("best epoch's test accuracy: %g"%(optimal_accuracy))
        print("---end---")
    


if __name__ == "__main__":
    stock=Stock()
    stock.train()

    



