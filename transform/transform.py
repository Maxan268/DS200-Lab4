from PIL import Image
import numpy as np
from skimage.feature import hog

def preprocess_image(image_path: str):
    """
    Đọc ảnh từ đường dẫn, resize về 64x64, chuyển ảnh về mức xám (grayscale),
    trích xuất đặc trưng HOG, chuẩn hóa và trả về danh sách đặc trưng.
    """
    # Mở ảnh và chuyển về ảnh xám
    img = Image.open(image_path).convert('L')
    # Resize ảnh về 64x64
    img = img.resize((64, 64))
    # Chuyển ảnh thành mảng numpy
    img_array = np.array(img, dtype=np.float32)
    # Chuẩn hóa giá trị pixel về khoảng [0,1]
    img_array /= 255.0
    # Trích xuất đặc trưng HOG
    features = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    # Trả về vector đặc trưng dưới dạng list
    return features.tolist()
