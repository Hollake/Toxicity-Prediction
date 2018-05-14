import tensorflow as tf 
import numpy as np 
import os
data_dir_train = "./NR-ER-train/names_onehots.npy"
labels_path_train = "./NR-ER-train/names_labels.txt"
data_dir_test = "./NR-ER-test/names_onehots.npy"
labels_path_test = "./NR-ER-test/names_labels.txt"
MODEL_SAVE_PATH="./model/"
MODEL_NAME="train_model"

# '''
#     labels_path:文件标签路径
#     data_dir:包含训练数据的onehots向量的.npy文件路径
#     return：
#     onehots_data：onehot数据列表,总共7697个数据，每个数据为72*398的onehot向量
#     molecule_name ：病毒名称列表
#     labels:病毒标签列表，有毒是1，无毒为0
# '''
def get_data(labels_path, data_dir):
	f = open(labels_path, 'r')
	contents = f.readlines()
	f.close()
	molecule_name = []
	labels = []
	for content in contents:
		value = content.split()
		name = value[0].split(',')[0]
		label = value[0].split(',')[1]
		molecule_name.append(name)
		labels.append(int(label))
	#从.npy文件中读出病毒数据
	dictionary = np.load(data_dir).item()
	onehots_data = dictionary['onehots']
	return  onehots_data, labels ,molecule_name

onehot_train, label_train, name_train = get_data(labels_path_train, data_dir_train)
onehot_test, label_test, name_test = get_data(labels_path_test, data_dir_test)

train_xdata = np.array(onehot_train)
test_xdata = np.array(onehot_test)
# print(train_xdata_or.shape)
# (7697, 72, 398)
train_labels = np.array(label_train)
test_labels = np.array(label_test)
# print(train_labels.shape)
# (7697,)
regularizer = 0.0001
batch_size = 100
learning_rate = 0.005
test_size = 100
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels)+1
num_channels = 1 
generations = 50
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 4 
max_pool_size2 = 4 
fully_connected_size1 = 100
#定义权重
def weight_variable(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w
#偏置函数
def bias_variable(shape):  
	b = tf.Variable(tf.zeros(shape))  
	return b
#最大池化，周边自动填充
def max_pool_4x4(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],strides=[1, 4, 4, 1], padding='SAME')
#卷积运算，
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))

test_input_shape = (test_size, image_width, image_height, num_channels)
test_input = tf.placeholder(tf.float32, shape=test_input_shape)
test_target = tf.placeholder(tf.int32, shape=(test_size))

global_step = tf.Variable(0, trainable=False) 
#定义前向传播算法
def forward(x, regularizer):
#第一卷积池化层
	W_conv1 = weight_variable([4, 4, num_channels, conv1_features],regularizer )
	b_conv1 = bias_variable([conv1_features])
	h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
	h_pool1 = max_pool_4x4(h_conv1)   
#第二卷积池化层
	w_conv2 = weight_variable([4,4,conv1_features,conv2_features],regularizer )
	b_conv2  = bias_variable([conv2_features])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
	h_pool2 = max_pool_4x4(h_conv2)
	#输出h_pool2的形状返回值为元组[5 25 40]
	final_conv_shape = h_pool2.get_shape().as_list()
	# print(final_conv_shape[1], final_conv_shape[2], final_conv_shape[3])
	#5 25 50
	final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
	#将h_pool2 reshape为bitch_size 大小的一维数组(100,5*25*50)
	flat_output = tf.reshape(h_pool2, [final_conv_shape[0], final_shape])
	# print(flat_output.shape)
	# (100, 6250)

#第一全连接层
	W_fc1 = weight_variable([final_shape, fully_connected_size1],regularizer )
	b_fc1 = bias_variable([fully_connected_size1])   
	h_fc1 = tf.nn.relu(tf.matmul(flat_output, W_fc1) + b_fc1)

#第二全连接层
	W_fc2 = weight_variable([fully_connected_size1, target_size],regularizer )
	b_fc2 = bias_variable([target_size])
	final_model_output=tf.matmul(h_fc1, W_fc2) + b_fc2

#返回预测结果
	return final_model_output
#定义准确率
def get_accuracy(logits, targets):
	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return(100*num_correct/batch_predictions.shape[0])

#获得预测结果
model_output = forward(x_input , regularizer)
test_model_output = forward(test_input ,regularizer)

prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)
#定义损失
cem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
#将L2正则化损失加入到loss中
loss = cem +tf.add_n(tf.get_collection('losses'))
#定义反向传播算法
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss,global_step=global_step)

train_loss = []
train_acc = []
test_acc = []
init = tf.global_variables_initializer()
#实例化saver
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	#如果指定路径下有模型则加载模型
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)

	# coord = tf.train.Coordinator()
	# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	for i in range(generations):
		#从len(train_xdata)中随机取出batch_size个角标
		rand_index = np.random.choice(len(train_xdata), size=batch_size)
		#返回取出随机角标所对应的数据形状为(100,72,389)
		rand_x = train_xdata[rand_index]
		#升维为(100,72,398,1)以符合输入数据要求
		rand_x = np.expand_dims(rand_x, 3)
		rand_y = train_labels[rand_index]
		train_dict = {x_input: rand_x, y_target: rand_y}
		
		sess.run(train_step, feed_dict=train_dict)
		temp_train_loss, temp_train_preds ,step = sess.run([loss, prediction , global_step], feed_dict=train_dict)
		temp_train_acc = get_accuracy(temp_train_preds, rand_y)
		
		if (i+1) % eval_every == 0:

			test_index = np.random.choice(len(test_xdata), size=test_size)
			test_x = test_xdata[test_index]
			test_x = np.expand_dims(test_x, 3)
			test_y = test_labels[test_index]
			test_dict = {test_input: test_x, test_target: test_y}
			test_preds = sess.run(test_prediction , feed_dict=test_dict)
			temp_test_acc = get_accuracy(test_preds, test_y)
			#保存训练模型
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
			print("Generation # %d. Train loss is %2f. Train accuracy is %2d.Test accuracy is %2d"
				%(i+1,temp_train_loss,temp_train_acc,temp_test_acc))
			# coord.request_stop()
			# coord.join(threads)
