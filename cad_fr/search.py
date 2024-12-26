import uuid
import cv2
from deepface import DeepFace

# 返回一个list，可能是空的list，[]
# 其中的元素格式为{"identity": xxx.jpg, "hash":xxx, “pos”:[x, y, w, h] }
# 其中的identity的路径是接着db_path的
# 如果db_path是c://xxx，那么identity就是c://xxx/xxx.jpg
# 如果db_path是xxx_db，那么identity就是xxx_db/xxx.jpg
def get_search_result(source_img_path, db_path, model_name, detector_backend="centerface", expand: int = 10):
    ret_list = []
    # 返回一个列表，列表的每个元素对应source_img中每个人脸的搜索结果
    search_result_list = DeepFace.find(
        img_path=source_img_path,
        db_path=db_path,
        enforce_detection=False,  # 让source里没有人脸，也不报错
        threshold=0.5,  # 过滤掉不相似图片的阈值
        detector_backend=detector_backend,
        model_name=model_name,
        expand_percentage=expand,
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
            # # # # 获取目标图片
            # tar_img = cv2.imread(tar_img_info["identity"])
            # print(tar_img_info["identity"])
            
            # # 加框
            # cv2.rectangle(tar_img, (tar_img_info["target_x"], tar_img_info["target_y"]),
            #               (tar_img_info["target_x"] + tar_img_info["target_w"],
            #                tar_img_info["target_y"] + tar_img_info["target_h"]),
            #               (0, 255, 0), 3)
            
            # # 展示目标图片
            # # tar_img = cv2.resize(tar_img, dsize=(800, 800))
            # print(tar_img.shape)
            # print(f'{tar_img_info["target_x"]}, {tar_img_info["target_y"]}')
            # cv2.imshow("source" + str(i) + "tar" + str(j), tar_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    return ret_list


if __name__ == "__main__":
    import sys
    import os
    
    param = sys.argv
    from cad_fr.config import settings
    
    if len(param) > 1:
        source_img_path = param[1]
        db_path = param[2]
    else:
        source_img_path = settings.demo_img_path
        db_path = settings.db_path

    model_name = settings.presentation_model_name
    detector_backend = settings.presentation_face_detector
    expand_percentage = settings.presentation_expand_percentage
    print(f"db_path: {db_path} model_name: {model_name} detector_backend: {detector_backend}")
    
    pkl_file_name = f"{db_path}/ds_model_{model_name.replace('-', '_').lower()}_detector_{detector_backend.lower()}_aligned_normalization_base_expand_{expand_percentage}.pkl"
    target_pkl_file_name = f"{db_path}/ds_model_{model_name.replace('-', '_').lower()}_detector_{detector_backend.lower()}_aligned_normalization_base_expand_{expand_percentage}.pkl"
    if os.path.exists(pkl_file_name):
        os.remove(pkl_file_name)
    
    # 搜索
    res = get_search_result(source_img_path, db_path, model_name, detector_backend, expand_percentage)
    print(res)


    import shutil
    shutil.copy(pkl_file_name, "config/ds_model.pkl")
    shutil.move(pkl_file_name, target_pkl_file_name)
