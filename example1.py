from defisheye import Defisheye


#img = "./192.168.10.41.jpg"
img = "./192.168.11.161.jpg"
img_out = "./result.jpg"

#obj = Defisheye(img, dtype='equalarea', xcenter=2094, ycenter=1465, radius=1400, pfov=120, p_radius=256)
obj = Defisheye(img, dtype='linear', xcenter=1500, ycenter=1500, radius=1500, pfov=140, p_radius=64)
obj.convert(img_out)


print(obj.convert_point(500, 1550))