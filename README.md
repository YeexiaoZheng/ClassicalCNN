# 实验运行方法

各模型的py文件中仅包含此模型，main.py为主文件，compare.py用于比较各个模型（已选取最优参数），plot.py中整合了画图，方便调用

## main.py

参数--model 选定模型

参数--lr 学习率 可以多填

参数--dropout 可以多填

参数--plot 是否画图

参数--epoch 轮次

**示例：**

```powershell
python .\main.py --model googlenet --lr 0.001 0.0005 0.0001 --dropout 0 0.2 0.4 --plot true --epoch 5
```

## compare.py

内置了五个经典模型以及合适的学习率，运行以下代码即可

```powershell
python .\compare.py
```

