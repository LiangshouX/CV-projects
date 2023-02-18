





## 1. Face Detection

### **Haar Cascade classifier**——The most common way to detect a face

Haar feature-based cascade classifiers ：

* A machine learning based approach where a cascade function is trained from a lot of positive and negative images

### How does it work？

* Training data：a lot of positive images (images of faces) and negative images (images without faces) 
*  Extract features from the classifier



*  Haar features shown in below image are used. Each feature is a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.

![image-20221122185209347](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221122185209347.png)

* Any problem?	-- Inefficient and time consuming to apply all the features to it
* The Resolution:   **Cascade of Classifiers**,  group the features into different stages of classifiers and apply one-by-one, the window which passes all stages is a face region

### Need Implement by Hand?

* No!  OpenCV has packaged all the process above with `CascadeClassifier`, just new an object
* Parameter: `haarcascade_frontalface_default.xml`

![image-20221123023416908](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123023416908.png)



If you do not want to create your own classifier, OpenCV already contains many pre-trained classifiers for face, eyes, smile, etc. Those XML files can be download from [haarcascades](https://github.com/Itseez/opencv/tree/master/data/haarcascades) directory.

`faceDetection.py`   are all you need to detect a face, using Python and OpenCV.

## 2. Data Gathering And Training

Facial Classifier `haarcascade_frontalface_default.xml`。

### Data Gethering

* Real-time capture from camera
  * capture 30 samples for each face id

* Dataset prepared
  * Didn't get suitable dataset yet
  * Gethered some tiny dataset for test
  * TODO

### Training

LBPH(Local Binary PatternsHistograms):	Be not affected by lighting, scaling, rotation, and translation.

![image-20221123005535810](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123005535810.png)



![image-20221123023149970](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123023149970.png)



Weight produced after training:	saved as `.yml` file, local appearance as below



![image-20221123012305412](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123012305412.png)

​	

Main Methods:	LBPHFaceRecognizer



## 3. Recognizer

Take videos, datasets containing images or camera frames as input,  the face recognizer (an object of LBPHFaceRecognizer) load the weight, then give the predicted face id and probability.



<img src="https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/New Flowchart.png" width=600>

![image-20221123030107658](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123030107658.png)

## 4. Results:



![image-20221123021619235](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123021619235.png)



![image-20221123022943021](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123022943021.png)

![image-20221123233229253](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123233229253.png)

![image-20221123233246379](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123233246379.png)

![image-20221123233313229](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123233313229.png)



![image-20221123233338980](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123233338980.png)



![image-20221123234601893](https://typora-1308640872.cos-website.ap-beijing.myqcloud.com/img/image-20221123234601893.png)

