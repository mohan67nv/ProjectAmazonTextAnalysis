import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mikkel import data_parser


class AmazonClassifier:
    def __init__(self, history_sampling_rate=1, w_init_limit=(-0.5, 0.5), display_step=1):
        self.history_sampling_rate = history_sampling_rate
        self.w_init_limit = w_init_limit
        self.display_step = display_step
        self.examples_to_show = 10

        self.one_hot_template = data_parser.load_bag_of_words()

        self.graph = tf.Graph()

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # tf placeholders
        self.X = None
        self.Y = None
        self.y_pred = None
        self.sess = None
        self.keep_prob = None  # for dropout
        self.saver = None

        # Network Parameters
        self.n_input = len(self.one_hot_template) + 1  # One-hot of most frequent words + 1 (unknown word)
        self.n_output = 5  # 5 stars to classify
        self.n_hidden_1 = 100  # 1st layer num features
        self.n_hidden_2 = 100  # 2nd layer num features

        self.cost_history = []
        self.test_acc_history = []

    def predict(self, reviews):
        all_results = []
        for review in reviews:
            words = data_parser.get_meaningful_words(review)
            one_hots = self.get_one_hot_from_words(words)
            results = self.sess.run(self.y_pred, feed_dict={self.X: one_hots, self.keep_prob: 1.0})
            print(results)
            print(np.mean(np.argmax(results, axis=1)))
            print(np.std(np.argmax(results, axis=1)))
            all_results.append(int(np.mean(np.argmax(results, axis=1)) + 1))
        return all_results

    def restore_model(self, restore_path='ann_model.ckpt'):
        self.build_model()
        self.saver.restore(self.sess, restore_path)
        print("Model restored from file: %s" % restore_path)

    def build_model(self, learning_rate=0.001):
        print("Building done graph...")
        with self.graph.as_default():
            # Encode curr_state, add transition prediction with selected action and decode to predicted output state
            self.X = tf.placeholder("float", [None, self.n_input])  # next state input
            self.Y = tf.placeholder("float", [None, self.n_output])  # output
            self.keep_prob = tf.placeholder(tf.float32)  # For dropout

            weights = {
                'h1': tf.Variable(
                    tf.random_normal([self.n_input, self.n_hidden_1])),
                'h2': tf.Variable(
                    tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'out': tf.Variable(
                    tf.random_normal([self.n_hidden_2, self.n_output])),
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'bout': tf.Variable(tf.random_normal([self.n_output])),
            }

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.X, weights['h1']), biases['b1']))
            layer_1 = tf.nn.dropout(layer_1, self.keep_prob)  # Dropout layer
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
            layer_2 = tf.nn.dropout(layer_2, self.keep_prob)  # Dropout layer
            out = tf.matmul(layer_2, weights['out']) + biases['bout']

            # Prediction
            self.y_pred = out
            # Targets (Labels) are the input data.
            y_true = self.Y

            delta = 0.0001
            self.loss_function = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.y_pred, y_true) + delta * tf.nn.l2_loss(
                    weights['h1']) + delta * tf.nn.l2_loss(weights['h2']) + delta * tf.nn.l2_loss(weights['out']))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_function)

            # Define loss, minimize the squared error (with or without scaling)
            # self.loss_function = tf.reduce_mean(tf.square(y_true - self.y_pred))
            # self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.01).minimize(self.loss_function)

            # Evaluate model
            # self.accuracy = tf.reduce_mean(tf.cast(self.loss_function, tf.float32))
            correct_prediction = tf.equal(tf.argmax(self.y_pred, 1), tf.argmax(y_true, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            # Creates a saver
            self.saver = tf.train.Saver()

            # Initializing the variables
            self.init = tf.initialize_all_variables()

            # Launch the graph
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(self.init)

    def generate_batch(self, batch_size, X_data, Y_data):
        idexes = np.arange(len(X_data))
        X_words = []
        Y_labels = []
        while len(X_words) < batch_size:
            idx = np.random.choice(idexes, 1, replace=True)
            review = X_data[idx][0]
            label = Y_data[idx][0] - 1  # 0-4, not 1-5 stars
            words = data_parser.get_meaningful_words(review)
            for word in words:
                one_hot = np.zeros(self.n_input)
                if word in self.one_hot_template:
                    one_hot[self.one_hot_template.index(word)] = 1.
                else:
                    one_hot[-1] = 1.
                X_words.append(one_hot)
                one_hot_label = np.zeros(self.n_output)
                one_hot_label[int(label)] = 1.
                Y_labels.append(one_hot_label)
        # TODO maybe return random indexes instead of the first ones
        return np.array(X_words[:batch_size]), np.array(Y_labels[:batch_size])

    def get_one_hot_from_words(self, words):
        X_words = []
        for word in words:
            one_hot = np.zeros(self.n_input)
            if word in self.one_hot_template:
                one_hot[self.one_hot_template.index(word)] = 1.
            else:
                one_hot[-1] = 1.
            X_words.append(one_hot)
        return np.array(X_words)

    def train(self, training_epochs=20, iterations_per_epoch=500, learning_rate=0.001, batch_size=128, show_cost=True,
              show_test_acc=True, save=False, save_path='done_model/tf_done_model.ckpt', logger=True):
        # Load and preprocess data
        if logger:
            print("Preprocessing data...")
        x_data, y_data = data_parser.load_training_data()

        # Split into training and testing data
        self.X_train = x_data[:1200000]
        self.Y_train = y_data[:1200000]
        self.X_test = x_data[1200000:]
        self.Y_test = y_data[1200000:]

        self.build_model(learning_rate=learning_rate)
        if logger:
            print("Starting training...")
            print("Total updates per batch:", iterations_per_epoch)
        # Training cycle
        for epoch in range(training_epochs):
            print("Starting epoch", epoch)
            # Loop over all batches
            c = None
            for i in range(iterations_per_epoch):
                batch_xs, batch_ys = self.generate_batch(batch_size, self.X_train, self.Y_train)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([self.optimizer, self.loss_function],
                                     feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: 1.0})
                if i % 10 == 0:
                    print("Processing batch", i, "cost:", c)
                if i % self.history_sampling_rate == 0:
                    self.cost_history.append(c)
                    batch_xs, batch_ys = self.generate_batch(256, self.X_test, self.Y_test)
                    self.test_acc_history.append(self.sess.run(self.accuracy,
                                                               feed_dict={self.X: batch_xs, self.Y: batch_ys,
                                                                          self.keep_prob: 1.0}))

            # Display logs per epoch step
            if epoch % self.display_step == 0 and c is not None:
                batch_xs, batch_ys = self.generate_batch(256, self.X_test, self.Y_test)
                test_error = self.sess.run(self.accuracy, feed_dict={self.X: batch_xs, self.Y: batch_ys,
                                                                     self.keep_prob: 1.0})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "test acc=",
                      "{:.9f}".format(test_error))

        batch_xs, batch_ys = self.generate_batch(10000, self.X_test, self.Y_test)
        final_acc = self.sess.run(self.accuracy,
                                  feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: 1.0})
        print("Final test acc:", final_acc)

        # Applying encode and decode over test set and show some examples
        batch_xs, batch_ys = self.generate_batch(self.examples_to_show, self.X_test, self.Y_test)
        prediction = self.sess.run(self.y_pred, feed_dict={self.X: batch_xs, self.keep_prob: 1.0})
        print(batch_ys[:self.examples_to_show])
        print(prediction[:])

        if save:
            save_path = self.saver.save(self.sess, save_path)
            print("Model saved in file: %s" % save_path)

        if show_test_acc:
            y_axis = np.array(self.test_acc_history)
            plt.plot(y_axis)
            plt.show()

        if show_cost:
            y_axis = np.array(self.cost_history)
            plt.plot(y_axis)
            plt.show()

        return final_acc


if __name__ == '__main__':
    clazzifier = AmazonClassifier()
    clazzifier.train(training_epochs=1, iterations_per_epoch=1000, learning_rate=0.001, batch_size=1024,
                     show_cost=False, show_test_acc=False, save=True, save_path='./tf_model.ckpt', logger=True)

    print("Testing")
    print(clazzifier.Y_test[:clazzifier.examples_to_show])
    clazzifier.predict([clazzifier.X_test[:clazzifier.examples_to_show]])
