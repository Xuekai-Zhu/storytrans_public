import os

def check_file_size(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024:  # 大于 100MB
                print(f"文件 {file_path} 大小超过 100MB")

# 指定要检查的文件夹路径
folder_path = "text_style_transfer/model"

# 调用函数进行文件大小检查
check_file_size(folder_path)
