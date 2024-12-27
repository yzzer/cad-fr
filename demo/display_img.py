from tkinter import Tk, Button, Label
from PIL import Image, ImageTk
import cv2
import numpy as np
from hashlib import sha256


# 模拟的人脸数据库，实际中你应该从你的真实数据来源替换此处
import pickle
with open('config/ds_model.pkl', 'rb') as f:
    face_database = pickle.load(f)
    face_database = sorted(face_database, key=lambda x: x["identity"])
    grouped_face_database = {}
    image_paths = []
    pre_identity = None
    pre_list = []
    for face in face_database:
        if pre_identity != face["identity"]:
            pre_list = []
            grouped_face_database[face["identity"]] = pre_list
            pre_identity = face["identity"]
            image_paths.append(pre_identity)
        pre_list.append(face)

current_index = 200  # 当前展示图片的索引

def hash_face(rep: dict) -> str:
    '''
    从人脸的特征向量中生成哈希值
    representation: np.ndarray, 人脸的特征向量
    return: str, 哈希值
    '''
    return sha256((f"{rep['hash']}_{rep['target_x']}_{rep['target_y']}_{rep['target_w']}_{rep['target_h']}").
            encode('utf-8')).hexdigest()

def draw_bboxes(image, reps):
    """在图片上绘制人脸方框以及对应的ID"""
    img_copy = image.copy()
    for rep in reps:
        x, y, w, h = rep["target_x"], rep["target_y"], rep["target_w"], rep["target_h"]
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, hash_face(rep), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return cv2.resize(img_copy, (2140, 1280), interpolation=cv2.INTER_AREA)


def show_next_image():
    global current_index
    current_index = (current_index + 1) % len(grouped_face_database)
    display_image()
    

def rm_face(hash):
    with open('config/ds_model.pkl', 'rb') as f:
        face_database = pickle.load(f)
    face_database = [face for face in face_database if hash_face(face) != hash]
    with open('config/ds_model.pkl', 'wb') as f:
        pickle.dump(face_database, f)
    print("rm success")



def display_image():
    global current_index
    image_path = image_paths[current_index]
    faces = grouped_face_database[image_path]

    image = Image.open(image_path)
    image_np = np.array(image)
    image_with_bboxes = draw_bboxes(image_np, faces)
    image_tk = ImageTk.PhotoImage(Image.fromarray(image_with_bboxes))

    image_label.config(image=image_tk)
    image_label.image = image_tk  # 保持引用，防止被垃圾回收
        
    while True:
        hash = input("hash: ")
        if hash == "n" or hash == "p":
            if current_index > len(image_paths) - 1:
                current_index = 0
                print("end")
            if hash == "p":
                current_index -= 2
            break
        for face in faces:
            hf = hash_face(face)
            if hf[:len(hash)] == hash and len(hash) > 0:
                rm_face(hf)
    show_next_image()

root = Tk()

# 创建显示图片的Label
image_label = Label(root)
image_label.pack()

display_image()

# 使用grid布局管理器
root.grid_rowconfigure(0, weight=1)  # 让第一行可伸缩，适应图片高度
root.grid_columnconfigure(0, weight=1)  # 让第一列可伸缩，适应图片宽度

# 创建显示图片的Label
image_label = Label(root)
image_label.grid(row=0, column=0, sticky="nsew")  # 占满整个第一行第一列

# 创建下一张按钮
next_button = Button(root, text="Next Image", command=show_next_image)
next_button.grid(row=1, column=0)  # 放置在第二行第一列

root.mainloop()