
准备资料：
修改读取预报数据的路径请在src/utils/models_util.py文件的第102行修改。
修改读取真实数据的路径请在src/utils/experiments_util.py文件的第291行修改。

进行训练：
训练脚本为onlinelearning_mod.py
gt_id：训练变量名（precip或tmp2m）（onlinelearning函数的var)
onlinelearning函数中的start_time和end_time为训练集的起始时间和结束时间
expert_models：参与训练的模型，输入范例为“p1，p2”或写有模型的txt文件。
alg：选择的算法：包括adahedged、dorm、dormplus、dub四种
hint：选择的提示方法：包括recent_g、mean_g、prev_g
注意：关于这四种算法以及三种提示方法的具体区别请参阅flaspohler的文献，online learning with optimism and delay。

lead_time：预报提前时间
target_dates：预报日期（注意：例如预报提前时间为10天，2020年1月1日报2020年1月11日，这种情况下1月11日是预报日期）

输出权重：
权重输出为一个numpy数组，可以选择将其保存在一个npy文件内以便后期使用，在onlinelearning_mod.py文件内可以更改保存地址以及文件名。
后期汇报文件的生成可以参考h5test内的文件，画图等脚本均有保存。
