## GSinit Exp
### Motivation
* 非`few-shot`版本的GS初始化点数在`10w`量级(`mip360`中的`kitchen`初始化点数是`241367`, `tank&temp`中的`train`是`182686`), 而`few-shot`的初始化点数只有`1k`量级(`kitchen`中大约是`8000`量级)
#### Exp 1
- few shot 和 non-few shot 结果对比
 - `kitchen`: Non-few shot: L1 0.0160362310 | PSNR 32.9537
### Method 1
- Use KNN to initialize Gaussian, based on GS distribution on 2D to guide 3D
### Method2
- using a 3D bounding box prediction and then using random initialization
  - Given several images and monodepth map and a 3D bbx prediction model, predict the 3D bbx. Initialize the points in the 3D bbx and refine the point cloud.
