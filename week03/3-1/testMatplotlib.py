import matplotlib.pyplot as plt
import numpy as np


def test():
    # 从[-1,1]中等距去50个数作为x的取值
    x = np.linspace(-1, 1, 50)
    print(x)
    y = 2*x + 1
    # 第一个是横坐标的值，第二个是纵坐标的值
    plt.plot(x, y)
    # 必要方法，用于将设置好的figure对象显示出来
    plt.show()

if __name__ == '__main__':
    test()
