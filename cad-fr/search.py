import uuid

import cv2
from deepface import DeepFace

# 返回一个list，可能是空的list，[]
# 其中的元素格式为{"identity": xxx.jpg, "hash":xxx, “pos”:[x, y, w, h] }
# 其中的identity的路径是接着db_path的
# 如果db_path是c://xxx，那么identity就是c://xxx/xxx.jpg
# 如果db_path是xxx_db，那么identity就是xxx_db/xxx.jpg
def get_search_result(source_img_path, db_path):
    ret_list = []
    # 返回一个列表，列表的每个元素对应source_img中每个人脸的搜索结果
    search_result_list = DeepFace.find(
        img_path=source_img_path,
        db_path=db_path,
        enforce_detection=False,  # 让source里没有人脸，也不报错
        threshold=0.5  # 过滤掉不相似图片的阈值
    )
    #
    for i, search_result in enumerate(search_result_list):
        for j, tar_img_info in search_result.iterrows():
            ret_list.append({
                "identity": tar_img_info["identity"],
                "hash": tar_img_info["hash"],
                "pos": [tar_img_info["target_x"], tar_img_info["target_y"]
                    , tar_img_info["target_w"], tar_img_info["target_h"]]
            })
            # # 获取目标图片
            # tar_img = cv2.imread(tar_img_info["identity"])
            #
            # # 加框
            # cv2.rectangle(tar_img, (tar_img_info["target_x"], tar_img_info["target_y"]),
            #               (tar_img_info["target_x"] + tar_img_info["target_w"],
            #                tar_img_info["target_y"] + tar_img_info["target_h"]),
            #               (0, 255, 0), 3)
            #
            # # 展示目标图片
            # tar_img = cv2.resize(tar_img, dsize=(800, 800))
            # cv2.imshow("source" + str(i) + "tar" + str(j), tar_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    return ret_list


# 需要去搜索的图片的路径
source_img_path = "source1.png"
# 去db_path下的照片中搜索，db_path下不能没有照片，不然报错
db_path = r"old_img_db"
# 搜索
res = get_search_result(source_img_path, db_path)
print(res)
