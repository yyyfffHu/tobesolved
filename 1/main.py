import cv2
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
img = cv2.imread('text.jpeg')

# def getDist(Point0,PointA):
#     distance=math.pow((Point0[0]-PointA[0]),2) + math.pow((Point0[1]-PointA[1]),2)
#     distance=math.sqrt(distance)
#     return distance
# l = int(getDist([313, 559], [991, 612]))
# w = int(getDist([991, 612], [995, 891]))

# l = 700
# w = 300

# def cv2_mat(pts1, pts2):
#     """
#     cv2源码mat计算
#     :param pts1:
#     :param pts2:
#     :return:
#     """
#     mat = cv.getPerspectiveTransform(pts1, pts2)
#     return mat

def self_mat(pts1, pts2):
    """
    A*Mat=B
    A  shape 8*8
    B  shape 8*1
    Mat shape 3*3 [a33]=1
    :param pts1:（4，2）
    :param pts2:（4，2）
    :return:
    """
    pts1 = pts1.reshape(4,2)
    pts2 = pts2.reshape(4,2)
    #定义A、B矩阵
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))
    for i in range(0, 4):
        A_i = pts1[i, :]
        B_i = pts2[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i ] = B_i[0]
        B[2 * i + 1] = B_i[1]
    #生产mat  计算warpMatric
    A = np.mat(A)
    Mat = A.I * B
    Mat = np.array(Mat).T[0]
    #插入a33=1
    Mat = np.insert(Mat, Mat.shape[0], values=1.0, axis=0)
    Mat = Mat.reshape((3, 3))
    return Mat



def perspective_main(src, dst, point_list=None):
    """
    image perspective transform
    :param src: images
    :param Mat: PerspectiveTransform Mat
    :param dst: sfz boxes
    :return:
    """
    img = src.copy()
    h, w, _ = img.shape
    src_point = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_point = np.float32(dst)

    perspective_mat = self_mat(dst_point, src_point)
    img_p = cv2.warpPerspective(img, perspective_mat, (w, h), borderValue=(0, 0, 0), flags=2)
    point_list_trans = np.zeros_like(point_list)
    for i, point in enumerate(point_list):
        t_point = []
        for j, poin in enumerate(point):
            u = poin[0]
            v = poin[1]
            x = int((perspective_mat[0][0] * u + perspective_mat[0][1] * v + perspective_mat[0][2]) /
                    (perspective_mat[2][0] * u + perspective_mat[2][1] * v + perspective_mat[2][2]))
            y = int((perspective_mat[1][0] * u + perspective_mat[1][1] * v + perspective_mat[1][2]) / \
                    (perspective_mat[2][0] * u + perspective_mat[2][1] * v + perspective_mat[2][2]))
            t_point.append([x, y])
        point_list_trans[i] = np.array(t_point)
    box_p = point_list_trans
    return img_p , box_p

# def perspective_res(img_p , box_p, mat):
#     img, box

if __name__ == "__main__":
    pts1 = np.float32([[313, 559], [991, 612], [995, 891], [290, 909]])
    # pts2 = np.float32([[0, 0], [l, 0], [l, w], [0, w]])
    # perspective_mat = cv2_mat(pts1, pts2)
    # perspective_mat_1 =self_mat(pts1, pts2)
    # dst = cv.warpPerspective(img, perspective_mat, (l, w), borderValue=(0,0,0),flags=2)
    point_list = np.array([378, 604, 472, 611, 459, 843, 360, 847]).reshape(-1, 4, 2)
    perspective_main(img, pts1, point_list)
# point_list = np.array([378, 604, 472, 611, 459, 843, 360, 847, 388, 275, 479, 293, 474, 507, 374, 502]).reshape(-1,4,2)





# point_list_trans = np.zeros_like(point_list)
# for i, point in enumerate(point_list):
#     t_point =[]
#     for j,poin in enumerate(point):
#
#         u = poin[0]
#         v = poin[1]
#         x = int((perspective_mat[0][0]*u+perspective_mat[0][1]*v+perspective_mat[0][2])/
#                 (perspective_mat[2][0]*u+perspective_mat[2][1]*v+perspective_mat[2][2]))
#         y = int((perspective_mat[1][0]*u+perspective_mat[1][1]*v+perspective_mat[1][2])/\
#                 (perspective_mat[2][0]*u+perspective_mat[2][1]*v+perspective_mat[2][2]))
#         t_point.append([x,y])
#     point_list_trans[i] =np.array(t_point)
#
#
# image_draw = dst.copy()
# for point_ in point_list_trans:
#     image_draw = cv2.polylines(image_draw, [point_list_trans.astype(np.int32).reshape((-1,1,2))],
#                                True,color= (255,0,0), thickness=2)
# # cv2.imwrite('1.jpg', image_draw)
# plt.subplot(121), plt.imshow(image_draw), plt.title('output')
# #
# pts = pts1.reshape(-1,4,2)
# image_draw= img.copy()
# for point_ in point_list:
#     image_draw = cv2.polylines(image_draw, [point_list.astype(np.int32).reshape((-1,1,2))],
#                                True,color= (0,0,255), thickness=5)
# for pts_ in pts:
#     image_draw = cv2.polylines(image_draw,[pts.astype(np.int32).reshape((-1,1,2))],
#                                True,color= (237,109,0), thickness=10)
# # cv2.imwrite('2.jpg', image_draw)
# plt.subplot(122), plt.imshow(image_draw), plt.title('input')
# plt.show()
