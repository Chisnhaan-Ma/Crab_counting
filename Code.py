import cv2 as CV
import numpy as np
import matplotlib.pyplot as plt  
from skimage.filters import threshold_multiotsu
import sys

def dilation(image, kernel_size):
    """
    Thực hiện phép giãn (dilation) trên ảnh nhị phân.
    
    Tham số:
    - image: Ảnh đầu vào (ảnh nhị phân).
    - kernel_size: Kích thước của kernel hình vuông.

    Trả về:
    - Ảnh sau khi thực hiện phép dilation.
    """
    # Tạo kernel hình vuông kích thước kernel_size x kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Tính padding để tránh lỗi biên
    pad = kernel_size // 2

    # Thêm padding vào ảnh (đệm 0)
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    # Tạo ảnh đầu ra
    dilated_image = np.zeros_like(image)

    # Duyệt qua từng pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Trích xuất vùng lân cận có kích thước bằng kernel
            region = padded_image[i:i+kernel_size, j:j+kernel_size]

            # Áp dụng toán tử dilation: pixel trung tâm nhận giá trị lớn nhất trong vùng
            dilated_image[i, j] = np.max(region)

    return dilated_image

def remove_small_large_objects(binary_image, min_size, max_size):
    """
    Loại bỏ các chấm quá nhỏ hoặc quá lớn trong ảnh nhị phân dựa trên diện tích.

    Args:
        binary_image (numpy.ndarray): Ảnh nhị phân đầu vào (0 và 255).
        min_area (int): Diện tích tối thiểu của một vùng để giữ lại.
        max_area (int): Diện tích tối đa của một vùng để giữ lại.

    Returns:
        numpy.ndarray: Ảnh sau khi loại bỏ các chấm theo diện tích.
    """
    # Đảm bảo ảnh là nhị phân với kiểu uint8 (giá trị 0 hoặc 255)
    binary_image = (binary_image > 128).astype(np.uint8) * 255

    # Tìm tất cả các thành phần liên thông
    num_labels, labels, stats, _ = CV.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Tạo ảnh kết quả
    filtered_image = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):  # Bỏ qua nhãn 0 (background)
        area = stats[i, CV.CC_STAT_AREA]
        if min_size <= area <= max_size:
            filtered_image[labels == i] = 255
        else :
            filtered_image[labels == i] = 0
    return filtered_image

def cat_da_nguong (image, T1, T2):
    m,n = image.shape
    image_thresh_hold = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            if (image[i,j] < T2):
                image_thresh_hold[i,j] = 0
            elif (T2 < image[i,j] < T1):
                image_thresh_hold[i,j] = 125
            else:
                image_thresh_hold[i,j] = 255
    return image_thresh_hold

def loc_trung_binh_co_trong_so(image):
    loc_TB = np.array(([1/9,1/9,1/9],
                       [1/9,1/9,1/9],
                       [1/9,1/9,1/9]), dtype="float")
    
    img_loc = convolution(image, loc_TB)
    return img_loc

def median_filter(image):
    m,n = image.shape
    img_filter = np.zeros([m,n])

    for i in range(1,m-1):
        for j in range(1,n-1):
            temp = [ image[i-1,j-1], 
                   + image[i-1,j], 
                   + image[i-1,j+1], 
                   + image[i,j-1],
                   + image[i,j],
                   + image[i,j+1], 
                   + image[i+1,j], 
                   + image[i+1,j-1], 
                   + image[i+1,j+1] ]
            temp = sorted(temp)
            img_filter[i,j] = temp[4]
    return img_filter

def erosion(image, kernel_size):
    # Tạo kernel hình vuông với kích thước kernel_size x kernel_size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Tính padding (để xử lý viền ảnh)
    pad = kernel_size // 2

    # Thêm padding vào ảnh (đệm 0)
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)

    # Tạo ảnh đầu ra
    eroded_image = np.zeros_like(image)

    # Duyệt qua từng pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Trích xuất vùng lân cận có kích thước bằng kernel
            region = padded_image[i:i+kernel_size, j:j+kernel_size]

            # Áp dụng toán tử erosion: pixel trung tâm chỉ giữ lại nếu toàn bộ kernel nằm trong vùng trắng
            eroded_image[i, j] = np.min(region)

    return eroded_image

def chuyen_doi_logarit(image,C):
    return float(C)*CV.log(1.0 + image)

def chuyen_doi_gamma(image, C, gamma):
    return float(C)*pow(image,float(gamma))

def dao_anh(image):
    return 255-image

def cat_nguong_thich_nghi(image, ksize):
    m,n = image.shape
    ket_qua = np.zeros([m,n])
    h = (ksize-1)//2 #mở rộng biên
    a = 3
    b = 0.5
    padded_img = np.pad(image,(h,h), mode = 'reflect')
    mG = np.mean(padded_img)
    for i in range(m):
        for j in range(n):
            vung_anh_kich_thuoc_k = padded_img[i:i+ksize, j: j+ksize]
            do_lech_chuan = np.std(vung_anh_kich_thuoc_k)
            T = a*do_lech_chuan + b*mG
            if (padded_img[i,j]> T):
                ket_qua[i,j] = 255
            else: 
                ket_qua[i,j] = 0

    return ket_qua

def tim_nguong_Otsu(image):
    vari_t = 0
    m,n = image.shape
    mG = np.mean(image)
    for th in range(256):
        gray_A = 1.0 
        gray_B = 1.0
        pixel_A = 1
        pixel_B = 1
        for i in range(m):
            for j in range(n):
                if (image[i,j]>= th):
                    pixel_A = pixel_A + 1
                    gray_A = gray_A + image[i,j]
                else:
                    pixel_B = pixel_B + 1
                    gray_B = gray_B + image[i,j]
                    
        P1 = pixel_A/(m*n)
        P2 = pixel_B/(m*n)
        m1 = gray_A/pixel_A
        m2 = gray_B/pixel_B
        vari = P1*((m1 - mG)**2) + P2*((m2 - mG)**2)

        if (vari > vari_t):
            vari_t = vari
            nguong_toi_uu = th
    print("Ngưỡng tìm được theo thuật toán Otsu:",nguong_toi_uu)
    return nguong_toi_uu

def thresh_hold(image, nguong):
    m,n = image.shape
    image_thresh_hold = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            if (image[i,j] < nguong):
                image_thresh_hold[i,j] = 0
            else:
                image_thresh_hold[i,j] = 255
    return image_thresh_hold

def Robert(image):
    Robert_filter_x = np.array(([0,0,0],
                                 [0,-1,0],
                                 [0,0,1]), dtype="float")

    Robert_filter_y = np.array(( [0,0,0],
                                [0,0,-1],
                                [0,1,0]), dtype="float")        

    Robert_x = convolution(image, Robert_filter_x)
    Robert_y = convolution(image, Robert_filter_y)
    return image + Robert_x +  Robert_y

def Sobel(image):
    Sobel_filter_x = np.array(([-1,-2,-1],
                                 [0,0,0],
                                 [1,2,1]), dtype="float")

    Sobel_filter_y = np.array(( [-1,0,1],
                                [-2,0,2],
                                [-1,0,1]), dtype="float")        

    Sobel_x = convolution(image, Sobel_filter_x)
    Sobel_y = convolution(image, Sobel_filter_y)
    return image + Sobel_x +  Sobel_y

def convolution(image,filter):
    m,n = image.shape
    img_filter = np.zeros([m,n])
    for i in range(1,m-1):
        for j in range(1,n-1):
            temp =   image[i-1,j-1] * filter[0,0]\
                   + image[i-1,j]   * filter[0,1]\
                   + image[i-1,j+1] * filter[0,2]\
                   + image[i,j-1]   * filter[1,0]\
                   + image[i,j]     * filter[1,1]\
                   + image[i,j+1]   * filter[1,2]\
                   + image[i+1,j]   * filter[2,0]\
                   + image[i+1,j-1] * filter[2,1]\
                   + image[i+1,j+1] * filter[2,2]
            img_filter[i,j] = temp
    img_filter = img_filter.astype(np.uint8)
    return img_filter

def convert_gray (image):
    gray_im = CV.cvtColor(image, CV.COLOR_BGR2GRAY)
    return gray_im

def image_processing(image):
    gray = convert_gray(image)

    im_1 = loc_trung_binh_co_trong_so(gray)

    im_2 = cat_nguong_thich_nghi(im_1, 10)

    im_3 = dao_anh(im_2)

    im_4 = erosion(im_3, 5)

    im_5 = dilation(im_4, 7)

    im_6 = erosion(im_5, 5)

    im_7 = remove_small_large_objects(im_6,10,5000)

    im_7 = (im_7 * 255).astype(np.uint8)

    contours, _ = CV.findContours(im_7, CV.RETR_EXTERNAL, CV.CHAIN_APPROX_SIMPLE)

# Vẽ contours lên ảnh gốc
    CV.drawContours(image, contours, -1, (0, 255, 0), 2)
    print(len(contours))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0,0].imshow(gray, cmap='gray')
    axes[0,0].set_title("Ảnh xám")
    axes[0,0].axis("off")   

    axes[0,1].imshow(im_3, cmap='gray')
    axes[0,1].set_title("Ảnh cắt ngưỡng thích nghi")
    axes[0,1].axis("off")    

    axes[0,2].imshow(im_4, cmap='gray')
    axes[0,2].set_title("Ảnh sau loại nhiễu")
    axes[0,2].axis("off")    


    axes[1,0].imshow(image, cmap='gray')
    axes[1,0].set_title("Ảnh Dilation cho mượt")
    axes[1,0].axis("off")
    

    axes[1,1].imshow(im_7, cmap='gray')
    axes[1,1].set_title("Ảnh lọc nhỏ lớn")
    axes[1,1].axis("off")

    axes[1,2].imshow(image, cmap='gray')
    axes[1,2].set_title("Ảnh erosion tách dính")
    axes[1,2].axis("off")
    
    #plt.show()
    

def main():

    image = CV.imread(r'D:\HCMUT\Year_2024-2025\242\DA1\data\z6455682814681_9337725b23daa97bf196e1a1b389aa38.jpg')

    image_processing(image)
    
    
if __name__ == "__main__":
    main()
    