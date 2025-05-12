import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


#-------------tmp2m------------

x = np.arange(1,30,1)
pred_rmse=np.load('./rmse_acc_2024/tmp2m_pred_rmse.npy')
ave_rmse=np.load('./rmse_acc_2024/tmp2m_ave_rmse.npy')
ecmf_rmse=np.load('./rmse_acc_2024/tmp2m_ecmf_rmse.npy')
cfs_rmse=np.load('./rmse_acc_2024/tmp2m_cfs_rmse.npy')
p0_rmse=np.load('./rmse_acc_2024/tmp2m_p0_rmse.npy')
p1_rmse=np.load('./rmse_acc_2024/tmp2m_p1_rmse.npy')
p2_rmse=np.load('./rmse_acc_2024/tmp2m_p2_rmse.npy')
p3_rmse=np.load('./rmse_acc_2024/tmp2m_p3_rmse.npy')

print(ecmf_rmse[14:30].mean(axis=0))
print(ave_rmse[14:30].mean(axis=0))
print(pred_rmse[14:30].mean(axis=0))


#----------原始程序------------
# plt.figure()

# plt.plot(x, p0_rmse,color='blue', linewidth=2, label='p0')
# plt.plot(x, p1_rmse,color='green', linewidth=2, label='p1')
# plt.plot(x, p2_rmse,color='purple', linewidth=2, label='p2')
# plt.plot(x, p3_rmse,color='orange', linewidth=2, label='p3')
# plt.plot(x, cfs_rmse,color='gray', linewidth=2, label='cfs')
# plt.plot(x, ecmf_rmse,color='brown', linewidth=2, label='ecmf')

# plt.plot(x, pred_rmse,color='red', linewidth=3, label='online')
# plt.plot(x, ave_rmse,color='black', linewidth=3, label='ave')

# #设置Legend
# plt.legend(loc='lower right')

# #标题
# plt.title('tmp2m')

# #添加文字
# plt.text(x=15,y=2.5, s='ecmf:3.71',fontsize=15)
# plt.text(x=15,y=3.0, s='pred:3.63',fontsize=15)
# plt.text(x=15,y=2.0, s='ave:3.66',fontsize=15)

# #显示图片
# plt.savefig("./tmp2m_rmse_20230101_20230430_all.png")
# plt.show()

#------------组图-------------

fig=plt.figure(figsize=(11,7))

ax1=fig.add_subplot(1,2,1)
plot1=ax1.plot(x, p0_rmse,color='blue', linewidth=2, label='p0')
plot1=ax1.plot(x, p1_rmse,color='green', linewidth=2, label='p1')
plot1=ax1.plot(x, p2_rmse,color='purple', linewidth=2, label='p2')
plot1=ax1.plot(x, p3_rmse,color='orange', linewidth=2, label='p3')
plot1=ax1.plot(x, cfs_rmse,color='gray', linewidth=2, label='cfs')

ax2=fig.add_subplot(1,2,2)
plot2=ax2.plot(x, ecmf_rmse,color='brown', linewidth=2, label='ecmf')
plot2=ax2.plot(x, ave_rmse,color='black', linewidth=2, label='ave')
plot2=ax2.plot(x, pred_rmse,color='red', linewidth=2, label='online')

#设置坐标标签
ax1.set_xlabel('days', fontsize=14)
ax1.set_xticklabels([-1,0,5,10,15,20,25,30],fontsize=16)
ax1.set_yticklabels([1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5],fontsize=16)

ax2.set_xlabel('days', fontsize=14)
ax2.set_xticklabels([-1,0,5,10,15,20,25,30],fontsize=16)
ax2.set_yticklabels([1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75],fontsize=16)

#设置Legend
ax1.legend(loc='lower right',fontsize=15)
ax2.legend(loc='lower right',fontsize=15)

#添加文字
ax2.text(x=18,y=3.25, s='ecmf:3.71',fontsize=18)
ax2.text(x=18,y=3.0, s='ave:3.66',fontsize=18)
ax2.text(x=18,y=2.75, s='online:3.63',fontsize=18)

#标题
plt.suptitle('tmp2m',fontsize=25)

#保存图像
plt.savefig("./tmp2m_rmse_20230101_20230430_all.png")
plt.show()

#-------------precip-------------

# x = np.arange(1,30,1)
# pred_rmse=np.load('./rmse_acc_2024/precip_pred_rmse.npy')
# ave_rmse=np.load('./rmse_acc_2024/precip_ave_rmse.npy')
# ecmf_rmse=np.load('./rmse_acc_2024/precip_ecmf_rmse.npy')
# cfs_rmse=np.load('./rmse_acc_2024/precip_cfs_rmse.npy')
# p0_rmse=np.load('./rmse_acc_2024/precip_p0_rmse.npy')
# p1_rmse=np.load('./rmse_acc_2024/precip_p1_rmse.npy')
# p2_rmse=np.load('./rmse_acc_2024/precip_p2_rmse.npy')
# p3_rmse=np.load('./rmse_acc_2024/precip_p3_rmse.npy')

# print(pred_rmse[14:30].mean(axis=0))
# print(ave_rmse[14:30].mean(axis=0))
# print(ecmf_rmse[14:30].mean(axis=0))

# fig=plt.figure(figsize=(11,7))

# ax1=fig.add_subplot(1,2,1)
# plot1=ax1.plot(x, p0_rmse,color='blue', linewidth=2, label='p0')
# plot1=ax1.plot(x, p1_rmse,color='green', linewidth=2, label='p1')
# plot1=ax1.plot(x, p2_rmse,color='purple', linewidth=2, label='p2')
# plot1=ax1.plot(x, p3_rmse,color='orange', linewidth=2, label='p3')
# plot1=ax1.plot(x, cfs_rmse,color='gray', linewidth=2, label='cfs')

# ax2=fig.add_subplot(1,2,2)
# plot2=ax2.plot(x, ecmf_rmse,color='brown', linewidth=2, label='ecmf')
# plot2=ax2.plot(x, ave_rmse,color='black', linewidth=2, label='ave')
# plot2=ax2.plot(x, pred_rmse,color='red', linewidth=2, label='online')

# #设置坐标标签
# ax1.set_xlabel('days', fontsize=14)
# ax1.set_xticklabels([-1,0,5,10,15,20,25,30],fontsize=16)
# ax1.set_yticklabels([4,6,8,10,12,14,16,18],fontsize=16)

# ax2.set_xlabel('days', fontsize=14)
# ax2.set_xticklabels([-1,0,5,10,15,20,25,30],fontsize=16)
# ax2.set_yticklabels([4,4.5,5.0,5.5,6.0,6.5,7.5],fontsize=16)

# #设置Legend
# ax1.legend(loc='lower right',fontsize=15)
# ax2.legend(loc='lower right',fontsize=15)

# #添加文字

# ax2.text(x=18,y=6.1, s='ecmf:6.77',fontsize=18)
# ax2.text(x=18,y=5.8, s='ave:6.74',fontsize=18)
# ax2.text(x=18,y=5.5, s='online:6.71',fontsize=18)

# #标题
# plt.suptitle('precip',fontsize=25)

# #保存图像
# plt.savefig("./precip_rmse_20230101_20230430_all.png")
# plt.show()

#------------双坐标轴-------------

# fig = plt.figure()

# axis_1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
# axis_2 = axis_1.twinx()

# # 绘制year-num数据集(折线图)
# # axis_1.plot(x, tp_grid_p0_rmse, color="#6AA84F", linewidth=2)
# axis_1.plot(x, p0_rmse,color='blue', linewidth=2, label='p0')
# axis_1.plot(x, p1_rmse,color='green', linewidth=2, label='p1')
# axis_1.plot(x, p2_rmse,color='purple', linewidth=2, label='p2')
# axis_1.plot(x, p3_rmse,color='orange', linewidth=2, label='p3')

# # 绘制year-loss数据集(条形图)
# axis_2.plot(x, cfs_rmse,color='gray', linewidth=2, label='cfs')
# axis_2.plot(x, ecmf_rmse,color='brown', linewidth=2, label='ecmf')
# axis_2.plot(x, ave_rmse, color="black", linewidth=2, label='ave')
# axis_2.plot(x, pred_rmse, color='red',  linewidth=2, label='online')
 
# # 设置坐标轴颜色
# plt.gca().spines["left"].set_color("black")
# plt.gca().spines["right"].set_color("red")
 
# # 设置坐标轴轴线宽度
# plt.gca().spines["left"].set_linewidth(2)
# plt.gca().spines["right"].set_linewidth(2)

# #设置X和Y坐标的范围
# axis_1.set_ylim(8, 17)
# axis_2.set_ylim(4, 12)

# #设置Legend
# axis_1.legend(loc='upper left',edgecolor = 'black')
# axis_2.legend(loc='lower right',edgecolor = 'red')

# #添加文字

# axis_1.text(x=15,y=9, s='ecmf:6.77',fontsize=15)
# axis_1.text(x=15,y=9.8, s='pred:6.71',fontsize=15)
# axis_1.text(x=15,y=8.2, s='ave:6.74',fontsize=15)

# #标题
# plt.title('precip')
# plt.savefig("./precip_rmse_20230101_20230430_all.png")
# plt.show()






# #设置X和Y坐标的范围
# ax1.set_xlim(-1, 10)
# ax1.set_ylim(0, 120)
#
# ax2.set_xlim(-1, 10)
# ax2.set_ylim(0, 60)
#
# #设置X轴和Y轴的名称
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
#
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
#
# #设置X轴和Y轴的坐标标签
# ax1.set_xticks(np.arange(0,11,2),[0,200,400,600,800,1000])
# ax2.set_xticks(np.arange(0,11,2),["a","b","c","d","e","f"])


