from defisheye import Defisheye


img = "./192.168.11.161.jpg"
img_out = "./result.jpg"

obj = Defisheye(img, dtype='equalarea', pfov=120, radius=256)
obj.convert(img_out)


print(obj.convert_point(1500, 1500))