# High Pass Skin Smoothing - Python


## Table of contents
1. [Notice](#notice)
2. [How to download a sample image](#get_image)
3. [Run](#run)
4. [Reference](#ref)
5. [Todo](#todo)


## 1. Notice <a name="notice"></a>
- I recommend that you should ignore the commented instructions with an octothorpe, #.
- This is implementation of [High Pass Skin Smoothing](https://www.google.com/search?ie=UTF-8&q=photoshop+high+pass+skin+smoothing) for Python3 which is inspired by [YUCIHighPassSkinSmoothing](https://github.com/YuAo/YUCIHighPassSkinSmoothing) and [HighPassSkinSmoothing-Android](https://github.com/msoftware/HighPassSkinSmoothing-Android).


## 2. How to download a sample image from [YUCIHighPassSkinSmoothing](https://github.com/YuAo/YUCIHighPassSkinSmoothing) <a name="get_image"></a>
```bash
wget https://raw.githubusercontent.com/YuAo/YUCIHighPassSkinSmoothing/refs/heads/master/YUCIHighPassSkinSmoothingDemo/YUCIHighPassSkinSmoothingDemo/Assets.xcassets/SampleImage.imageset/SampleImage.jpg -O ./asset/SampleImage.jpg
```


## 3. How to run <a name="run"></a>
```bash
$ python3 main.py --path_img_src "./asset/SampleImage.jpg" --path_img_dst "./asset/SampleImage_refined.jpg" --level_auto
$ python3 main.py --path_img_src "./asset/SampleImage.jpg" --path_img_dst "./asset/SampleImage_refined.jpg" --level_smooth "200.0" --level_whiten "3.0"
```


## 4. Previews from [YUCIHighPassSkinSmoothing](https://github.com/YuAo/YUCIHighPassSkinSmoothing)
![Preview 1](http://yuao.github.io/YUCIHighPassSkinSmoothing/previews/1.jpg)
![Preview 2](http://yuao.github.io/YUCIHighPassSkinSmoothing/previews/2.jpg)
![Preview 3](http://yuao.github.io/YUCIHighPassSkinSmoothing/previews/3.jpg)
![Preview 4](http://yuao.github.io/YUCIHighPassSkinSmoothing/previews/4.jpg)
![Preview 5](http://yuao.github.io/YUCIHighPassSkinSmoothing/previews/5.jpg)
![Preview 6](http://yuao.github.io/YUCIHighPassSkinSmoothing/previews/6.jpg)


## 5. Reference <a name="ref"></a>
1. [YUCIHighPassSkinSmoothing](https://github.com/YuAo/YUCIHighPassSkinSmoothing)
2. [HighPassSkinSmoothing-Android](https://github.com/msoftware/HighPassSkinSmoothing-Android)


## 6. Todo <a name="todo"></a>
