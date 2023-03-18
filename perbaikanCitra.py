# F55121006_Felisya Brigita Tania Wong_A

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# fungsi mengubah gambar ke grayscale
def grayscale(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

# fungsi untuk menerapkan transformasi citra negatif
def negative_transform(img):
    gray_img = grayscale(img)
    neg_img = cv2.bitwise_not(gray_img)
    return neg_img

# fungsi untuk melakukan perataan citra
def image_averaging(img):
    average_image = np.zeros_like(img[0], dtype=np.float32)
    for image in img:
        average_image += image.astype(np.float32)
    average_image /= len(img)
    return np.uint8(average_image)

# fungsi untuk menerapkan filter rata-rata pada citra
def average_filter(img):
    kernel_size = 21
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

# fungsi untuk menerapkan filter median pada citra
def median_filter(img):
    kernel_size = 5
    median_filter_img = cv2.medianBlur(img, kernel_size)
    return median_filter_img

# fungsi untuk menerapkan filter minimum pada citra (digunakan untuk mengurangi noise atau derau pada citra)
def min_filter(img):
    kernel = np.ones((3, 3), np.uint8)
    min_filter_img = cv2.erode(img, kernel)
    return min_filter_img

# fungsi untuk menghaluskan gambar
def gaussian_lowpass_filter(img):
    kernel_size = 21
    sigma = 5
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_lowpass_filter_img = cv2.filter2D(img, -1, kernel)
    return gaussian_lowpass_filter_img

# fungsi untuk meningkatkan ketajaman citra
def unsharp_masking(img):
    blurred_img = cv2.GaussianBlur(img, (0, 0), 2)
    high_freq = img.astype(np.float32) - blurred_img.astype(np.float32)
    sharpened_img = img.astype(np.float32) + 1.5 * high_freq
    sharpened_img = np.clip(sharpened_img, 0, 255)
    sharpened_img = np.uint8(sharpened_img)
    return sharpened_img

# fungsi untuk melakukan filterisasi pada citra dengan menggunakan pendekatan selektif
def selective_filtering(img):
    # konversi citra ke grayscale jika citra berwarna
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # periksa ukuran citra
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("Ukuran citra tidak valid")
    # periksa tipe data citra
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = np.float32(img)
    # konversi citra ke domain frekuensi
    img_dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    img_dft_shifted = np.fft.fftshift(img_dft)
    # buat filter selektif berbentuk lingkaran dengan jari-jari tertentu
    rows, cols = img.shape
    radius = 30
    mask = np.zeros((rows, cols, 2), np.float32)
    center = (cols//2, rows//2)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    mask[mask_area] = 1
    # terapkan filter selektif pada citra di domain frekuensi
    filtered_dft = img_dft_shifted * mask
    filtered_dft_shifted = np.fft.ifftshift(filtered_dft)
    filtered_img = cv2.idft(filtered_dft_shifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    # normalisasi nilai pixel pada citra hasil
    filtered_img_norm = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return filtered_img_norm

# fungsi untuk mengkonversi citra RGB ke HSV (Hue, Saturation, Value)
def rgb_to_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_img

# fungsi untuk meningkatkan kontras gambar dengan memperluas rentang kecerahan gambar
def histogram_equalization(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_img = cv2.equalizeHist(gray_img)
    return eq_img

# fungsi untuk meningkatkan tepi pada citra
def edge_enhancement(img):
    gray_img = grayscale(img)
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    edge_img = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    return edge_img

# fungsi untuk menyesuaikan kecerahan gambar
def brightness_adjustment(img, alpha=1.5, beta=10):
    brightened_img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
    return brightened_img

# fungsi untuk menghilangkan blur pada citra
def deblurring(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blurred_img = cv2.filter2D(img, -1, kernel)
    fft_blurred_img = np.fft.fft2(blurred_img)
    fft_kernel = np.fft.fft2(kernel, s=img.shape[:2])
    fft_kernel = np.stack([fft_kernel] * 3, axis=-1)
    fft_deblurred_img = np.divide(fft_blurred_img, fft_kernel)
    deblurred_img = np.real(np.fft.ifft2(fft_deblurred_img))
    deblurred_img = np.clip(deblurred_img, 0, 255).astype(np.uint8)
    return deblurred_img

# fungsi untuk menampilkan gambar dalam kotak
def show_image(img, x, y, title):
    img = cv2.resize(img, (150, 150))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=img)
    label.image = img
    label.place(x=x, y=y)
    title_label = tk.Label(root, text=title)
    title_label.place(x=x, y=y-20)

# fungsi untuk memproses citra dan menampilkan hasilnya
def process_image(method):
    global original_img
    if method == 'grayscale':
        corrected_img = grayscale(original_img)
        show_image(corrected_img, 458, 40, 'Metode Perbaikan 1')
    elif method == 'negative_transform':
        corrected_img = negative_transform(original_img)
        show_image(corrected_img, 698, 40, 'Metode Perbaikan 2')
    elif method == 'image_averaging':
        corrected_img = image_averaging(original_img)
        show_image(corrected_img, 938, 40, 'Metode Perbaikan 3')
    elif method == 'average_filter':
        corrected_img = average_filter(original_img)
        show_image(corrected_img, 1178, 40, 'Metode Perbaikan 4')
    elif method == 'median_filter':
        corrected_img = median_filter(original_img)
        show_image(corrected_img, 218, 265, 'Metode Perbaikan 5')
    elif method == 'min_filter':
        corrected_img = min_filter(original_img)
        show_image(corrected_img, 458, 265, 'Metode Perbaikan 6')
    elif method == 'gaussian_lowpass_filter':
        corrected_img = gaussian_lowpass_filter(original_img)
        show_image(corrected_img, 698, 265, 'Metode Perbaikan 7')
    elif method == 'unsharp_masking':
        corrected_img = unsharp_masking(original_img)
        show_image(corrected_img, 938, 265, 'Metode Perbaikan 8')
    elif method == 'selective_filtering':
        corrected_img = selective_filtering(original_img)
        show_image(corrected_img, 1178, 265, 'Metode Perbaikan 9')
    elif method == 'rgb_to_hsv':
        corrected_img = rgb_to_hsv(original_img)
        show_image(corrected_img, 218, 485, 'Metode Perbaikan 10')
    elif method == 'histogram_equalization':
        corrected_img = histogram_equalization(original_img)
        show_image(corrected_img, 458, 485, 'Metode Perbaikan 11')
    elif method == 'edge_enhancement':
        corrected_img = edge_enhancement(original_img)
        show_image(corrected_img, 698, 485, 'Metode Perbaikan 12')
    elif method == 'brightness_adjustment':
        corrected_img = brightness_adjustment(original_img)
        show_image(corrected_img, 938, 485, 'Metode Perbaikan 13')
    elif method == 'deblurring':
        corrected_img = deblurring(original_img)
        show_image(corrected_img, 1178, 485, 'Metode Perbaikan 14')

# fungsi untuk menampilkan informasi pembuat program
def show_creator():
    creator_label = tk.Label(root, text='Nama : Felisya Brigita Tania Wong                                                                                                                                                                 '
                                        'NIM : F55121006                                                                                                                                                                 '
                                        'Kelas : A', anchor='w')
    creator_label.place(x=40, y=685)

# fungsi untuk membuka gambar
def open_image():
    global original_img
    file_path = filedialog.askopenfilename()
    if file_path:
        original_img = cv2.imread(file_path)
        show_image(original_img, 218, 40, 'Original Image')

# membuat jendela utama
root = tk.Tk()
root.geometry('1500x1000')
root.title('GUI Aplikasi Penerapan Perbaikan Citra')

# menambahkan kotak untuk perbaikan citra
correction_box = tk.LabelFrame(root, text='Perbaikan Citra', padx=5, pady=5)
correction_box.place(x=20, y=20, width=170, height=620)

# tombol untuk membuka gambar
open_button = tk.Button(correction_box, text='Select an Image', command=open_image)
open_button.pack(side=tk.TOP, padx=5, pady=10)

# tombol untuk perbaikan metode 1 (grayscaling)
smoothing_button = tk.Button(correction_box, text='1. Grayscaling', command=lambda: process_image('grayscale'))
smoothing_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 2 (negative_transform)
negative_transform_button = tk.Button(correction_box, text='2. Negative Transform', command=lambda: process_image('negative_transform'))
negative_transform_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 3 (image_averaging)
image_averaging_button = tk.Button(correction_box, text='3. Image Averaging', command=lambda: process_image('image_averaging'))
image_averaging_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 4 (average_filter)
average_filter_button = tk.Button(correction_box, text='4. Average Filter', command=lambda: process_image('average_filter'))
average_filter_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 5 (median_filter)
median_filter_button = tk.Button(correction_box, text='5. Median Filter', command=lambda: process_image('median_filter'))
median_filter_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 6 (min_filter)
min_filter_button = tk.Button(correction_box, text='6. Min Filter', command=lambda: process_image('min_filter'))
min_filter_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 7 (gaussian_lowpass_filter)
gaussian_lowpass_filter_button = tk.Button(correction_box, text='7. Gaussian Lowpass Filter', command=lambda: process_image('gaussian_lowpass_filter'))
gaussian_lowpass_filter_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 8 (unsharp_masking)
unsharp_masking_button = tk.Button(correction_box, text='8. Unsharp Masking', command=lambda: process_image('unsharp_masking'))
unsharp_masking_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 9 (selective_filtering)
selective_filtering_button = tk.Button(correction_box, text='9. Selective Filtering', command=lambda: process_image('selective_filtering'))
selective_filtering_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 10 (rgb_to_hsv)
rgb_to_hsv_button = tk.Button(correction_box, text='10. RGB to HSV', command=lambda: process_image('rgb_to_hsv'))
rgb_to_hsv_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 11 (histogram_equalization)
histogram_equalization_button = tk.Button(correction_box, text='11. Histogram Equalization', command=lambda: process_image('histogram_equalization'))
histogram_equalization_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 12 (edge_enhancement)
edge_enhancement_button = tk.Button(correction_box, text='12. Edge Enhancement', command=lambda: process_image('edge_enhancement'))
edge_enhancement_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 13 (brightness_adjustment)
brightness_adjustment_button = tk.Button(correction_box, text='13. Brightness Adjustment', command=lambda: process_image('brightness_adjustment'))
brightness_adjustment_button.pack(side=tk.TOP, padx=5, pady=6)

# tombol untuk perbaikan metode 14 (deblurring)
deblurring_button = tk.Button(correction_box, text='14. Deblurring', command=lambda: process_image('deblurring'))
deblurring_button.pack(side=tk.TOP, padx=5, pady=6)

# menambahkan kotak untuk informasi pembuat program
creator_box = tk.LabelFrame(root, text='Creator', padx=5, pady=5)
creator_box.place(x=20, y=660, width=1325, height=65)

# menampilkan informasi pembuat program
show_creator()

# menjalankan program
root.mainloop()
