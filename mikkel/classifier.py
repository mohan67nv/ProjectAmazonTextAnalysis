import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

from mikkel import data_parser

"""
TODO:
- Try normalizing the biased data
- Try uncertain value (distribution over the different stars)
- Try longer training time
- Try other network configurations
-

If nothing works (new ideas):
- Try LSTMs
- Try regression instead of classification
"""


class AmazonClassifier:
    def __init__(self, history_sampling_rate=1, w_init_limit=(-0.5, 0.5), display_step=1):
        self.history_sampling_rate = history_sampling_rate
        self.w_init_limit = w_init_limit
        self.display_step = display_step
        self.examples_to_show = 10

        self.one_hot_template = np.array(pickle.load(open("bag_of_words.pickle", "rb")))

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
        self.n_input = len(self.one_hot_template)  # + 1  # One-hot of most frequent words + 1 (unknown word)
        self.n_output = 5  # 5 stars to classify
        self.n_hidden_1 = 500  # 1st layer num features
        self.n_hidden_2 = 500  # 2nd layer num features

        self.cost_history = []
        self.test_acc_history = []

        self.star_counter = np.zeros(5)

    def predict(self, reviews, much_ram_needed=True):
        all_results = []

        if much_ram_needed:
            if len(reviews) > 10000:
                while len(all_results) < len(reviews):
                    batch_xs, batch_ys = self.generate_batch(None, reviews, None, index_from=len(all_results),
                                                             index_to=len(all_results) + min(10000, len(reviews) - len(
                                                                 all_results)))
                    results = self.sess.run(self.y_pred, feed_dict={self.X: batch_xs, self.keep_prob: 1.0})
                    for res in results:
                        all_results.append(np.argmax(res) + 1)
            else:
                batch_xs, batch_ys = self.generate_batch(min(10000, len(reviews)), reviews, None)
                results = self.sess.run(self.y_pred, feed_dict={self.X: batch_xs, self.keep_prob: 1.0})
                for res in results:
                    all_results.append(np.argmax(res) + 1)
        else:
            for review in reviews:
                if len(review) > 0:
                    words = data_parser.get_meaningful_words(review)
                    one_hots = self.get_one_hot_from_words(words)
                    results = self.sess.run(self.y_pred, feed_dict={self.X: [one_hots], self.keep_prob: 1.0})
                    all_results.append(round(np.mean(np.argmax(results, axis=1)) + 1))
                else:
                    print("Something went wrong in the prediction...")
                    all_results.append(5)
        return all_results

    def restore_model(self, restore_path='./tf_model.ckpt'):
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
            # out = tf.matmul(layer_2, weights['out']) + biases['bout']
            out = tf.add(tf.matmul(layer_2, weights['out']), biases['bout'])
            # sfm = tf.nn.softmax(tf.nn.tanh(out))

            # Prediction
            self.y_pred = out
            # Targets (Labels) are the input data.
            y_true = self.Y

            delta = 0.0001
            self.loss_function = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(self.y_pred, y_true))
            # + delta * tf.nn.l2_loss(
            #         weights['h1']) + delta * tf.nn.l2_loss(weights['h2']) + delta * tf.nn.l2_loss(weights['out']))
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
            self.init = tf.global_variables_initializer()

            # Launch the graph
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.graph, config=config)
            self.sess.run(self.init)

    def generate_batch(self, batch_size, X_data, Y_data, index_from=None, index_to=None):
        idexes = np.arange(len(X_data))
        X_words = []
        Y_labels = []
        if index_from is not None and index_to is not None:
            chosen_indexes = np.arange(index_from, index_to)
        else:
            chosen_indexes = np.random.choice(idexes, batch_size)
        # selected_label = np.argmin(self.star_counter)
        for idx in np.nditer(chosen_indexes):
            if Y_data is not None:
                label = Y_data[idx] - 1.  # 0-4, not 1-5 stars
                one_hot_label = np.zeros(self.n_output)
                one_hot_label[int(label)] = 1.
                Y_labels.append(one_hot_label)
            # while label != selected_label:
            #     idx = np.random.choice(idexes, 1, replace=True)
            #     label = Y_data[idx][0] - 1  # 0-4, not 1-5 stars
            # self.star_counter[selected_label] += 1
            review = X_data[idx]
            words = data_parser.get_meaningful_words(review)
            X_words.append(self.get_one_hot_from_words(words))
        return np.array(X_words), np.array(Y_labels)

    def get_one_hot_from_words(self, words):
        # prev_idx = None
        one_hot = np.zeros(self.n_input)
        for word in words:
            if word in self.one_hot_template:
                idx = np.where(self.one_hot_template == word)
                # if prev_idx is not None:
                #     one_hot[prev_idx] = 1.
                one_hot[idx] = 1.
                # prev_idx = idx
                # else:
                #     one_hot[-1] = 1.
        return one_hot

    def load_data(self):
        # Load and preprocess data
        x_data, y_data = data_parser.load_training_data()

        # Split into training and testing data
        self.X_train = x_data[:1200000]
        self.Y_train = y_data[:1200000]
        self.X_test = x_data[1200000:]
        self.Y_test = y_data[1200000:]
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def train(self, training_epochs=20, iterations_per_epoch=500, learning_rate=0.001, batch_size=128, show_cost=True,
              show_test_acc=True, save=False, save_path='done_model/tf_done_model.ckpt', logger=True):
        if self.X_train is None:
            if logger:
                print("Preprocessing data...")
            self.load_data()

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
                                     feed_dict={self.X: batch_xs, self.Y: batch_ys, self.keep_prob: 1.})
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
                if save:
                    self.saver.save(self.sess, save_path)

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
    super_start_time = time.time()
    save_path = './model_saves/tf_model.ckpt'
    clazzifier = AmazonClassifier(history_sampling_rate=10)
    X_data, Y_data, X_test, Y_test = clazzifier.load_data()
    data_parser.generate_index_representation(X_data, Y_data, save_to_pickle=True, max_data=None)
    # clazzifier.train(training_epochs=10, iterations_per_epoch=500, learning_rate=0.001, batch_size=64,
    #                  show_cost=False, show_test_acc=True, save=True, save_path=save_path, logger=True)
    # # clazzifier.restore_model(restore_path=save_path)
    # clazzifier.load_data()
    # start_time = time.time()
    # print("Testing")
    # test_n_samples = len(clazzifier.Y_test)
    # print(clazzifier.Y_test[:10])
    # print(clazzifier.predict(clazzifier.X_test[:10]))
    # res_true = np.array(clazzifier.Y_test[:test_n_samples])
    # # res_pred = np.zeros(test_n_samples)
    # # res_pred.fill(5.)
    # res_pred = np.array(clazzifier.predict(np.array(clazzifier.X_test[:test_n_samples]), much_ram_needed=True))
    # acc = np.sum(res_true == res_pred) / test_n_samples
    # mse = ((res_true - res_pred) ** 2).mean()
    #
    # print("Acc:", acc)
    # print("MSE:", mse)
    # print("Prediction time used:", time.time() - start_time)
    print("All time used:", time.time() - super_start_time)

    # data_parser.draw_heatmap(res_true, res_pred)
