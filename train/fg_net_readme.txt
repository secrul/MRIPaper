1.损失函数：
    辅助损失、主损失，gamma为0.01；对比只要主损失较优
    损失：辅助和主损失都是k空间+img
          辅助为img、主损失为k空间+img
          辅助为img且guass、主损失为k空间+img
          辅助为img且guass、主损失为img，2c
          辅助为img且guass、主损失为img，1c
          辅助为img且guass、主损失为img，1c
          损失都为img，辅助guass，加ssim

    模型：膨胀卷积、密集连接
          

消融：
      gamma为0.01
      gamma为1
      gamma为0
      去掉子网络，级联block加长，参数不变
