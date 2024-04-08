import os
import cv2
import random
interpolation_method=[cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_LINEAR]#插值方法

 
hr_image_dir = "../../SID_output_test_3072x2048/NOISE_weibo"
lr_image_dir = hr_image_dir+"_lr"
 
img_size=(384,256)#img_size
# 判断输入目录是否存在
if not os.path.exists(hr_image_dir):
    print(f"directory {hr_image_dir} does not exist.")
    exit()
target_dir=lr_image_dir + "/X8"#存储目录
if not os.path.exists(target_dir):
    #create LR image dirs
    os.makedirs(target_dir)
 
supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".PNG",".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")
 
#Downsample HR images
for filename in os.listdir(hr_image_dir):
    flag=random.randint(0,2)#随机选择插值方法
    if not filename.endswith(supported_img_formats):
        print(f"img{filename} processing fail")
        continue
    name, ext = os.path.splitext(filename) 
    #Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])
    #Downsample image 
    print(f"method:{interpolation_method[flag]}")
    lr_image = cv2.resize(hr_img,dsize=img_size, interpolation=interpolation_method[flag])
    # if args.keepdims:
    #     lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=interpolation_method[flag])
 
    cv2.imwrite(os.path.join(target_dir, name+'_x8'+ext), lr_image)
