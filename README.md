# Toxicity-Prediction\<br>  
毒性检测\<br>  
项目的详细说明以及要求DNN_Project_Specification_new_send.pdf文件里有详细说明.\<br>  
数据在NR-ER-test和NR-ER-test两个文件夹里，可以读取.npy的one_hot向量形式的数组数据，也有可以读取SMILE notion用RNN实现。\<br>   
数据在https://pan.baidu.com/s/1AoXmNrbzgKXdMp_fVk6M8A 自取，在运行程序时将下载的NR-ER-test和NR-ER-test放在和run.py同一级目录下\<br>  
我采用的是one_hot向量模型输入\<br> 
我所实现的CNN神经网络训练50轮可以达到90%的准确率，相关超参数的调试没有详细进行，有兴趣的同学可以进行调试\<br>  
#问题\<br>  
卷积核参考其他模型用的是4*4大小，因为发现3*3和5*5训练准确率和4*4的没有多大区别，但是测试准确率差距能达到70%，这一点我不是很明白\<br>  
想在每次训练时保存模型，在下次训练时恢复模型，但是模型可以保存，下次训练时直接会覆盖模型，并不能恢复模型继续进行训练\<br>  
