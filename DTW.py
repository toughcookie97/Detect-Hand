import math
import numpy as np
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
#模板类分别是上挥手、下挥手、左挥手、右挥\旋转1、旋转2
modules=[
            [4, 0, 4, 0, 0, 0, 3, 7, 4, 0, 5, 0, 3, 3, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 3],
    [0, 7, 4, 5, 2, 0, 0, 7, 6, 4, 7, 5, 7, 5, 5, 6, 6, 6, 6, 7, 6, 6, 6, 6, 5, 6, 4, 7, 6, 6, 6, 2, 5, 6, 5, 5, 6],
    [7, 1, 0, 0, 0, 1, 0, 0, 0, 0, 7, 0, 0, 0, 7, 7, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 7, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0,
     7, 0, 0, 7],
    [1, 4, 2, 6, 6, 2, 6, 3, 4, 5, 6, 1, 1, 5, 3, 4, 3, 3, 3, 5, 3, 3, 5, 3, 4, 4, 4, 4, 5, 4, 3, 5, 3, 2],
[4, 1, 0, 0, 2, 4, 7, 0, 5, 2, 3, 3, 4, 5, 3, 4, 4, 4, 4, 5, 4, 7, 0, 1, 0, 7, 2, 0, 0, 0, 0, 0, 7, 5, 3, 2, 5, 4, 3, 4, 4, 4, 5, 4, 4, 4, 4, 4, 5, 0, 0, 0, 0, 7, 0],            #[1,2,3,4,5,6,7,0]
         ]
moduleNames=["上挥手","下挥手","左挥手","右挥手","旋转1","旋转2"]
def judgeGesture(points):
    # 将手部某个特征点在各个帧的坐标存放到这个数组即可
    # 将手势的走向分成八个方向


    basicPoints = [[1, 0], [math.sqrt(2) / 2, -math.sqrt(2) / 2], [0, -1], [-math.sqrt(2) / 2, -math.sqrt(2) / 2],
                   [-1, 0], [-math.sqrt(2) / 2, math.sqrt(2) / 2], [0, 1], [math.sqrt(2) / 2, math.sqrt(2) / 2]]

    vectors = []
    for i in range(1, len(points)):
        vectors.append([points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]])

    res = [[0 for col in range(len(basicPoints))] for row in range(len(points))]

    for i in range(0, len(vectors)):
        for j in range(0, len(basicPoints)):
            res[i][j] = vectors[i][0] * basicPoints[j][0] + vectors[i][1] * basicPoints[j][1]

    vectorRes = []
    for i in range(0, len(vectors)):
        vectorRes.append(np.argmax(res[i]))
    print(vectorRes)

    distances = []
    for i in range(0, len(modules)):
        distances.append(dtw.distance(modules[i], vectorRes))
    minIndex = distances.index((min(distances)))

    print(moduleNames[minIndex])
"""
if __name__ == '__main__':
    poinssss=points = [[74.0, 243.0], [79.5, 231.5], [81.0, 224.5], [82.5, 224.5], [80.0, 227.5], [97.5, 219.0], [107.0, 215.5],
              [135.5, 216.0], [133.5, 210.0], [154.5, 210.5], [253.0, 249.0], [174.5, 213.0], [191.0, 214.5],
              [205.0, 212.5], [216.0, 202.5], [234.0, 205.5], [253.0, 196.5], [272.5, 195.5]]
    judgeGesture(points)
"""





