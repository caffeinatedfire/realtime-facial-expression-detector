# realtime-facial-expression-detector
Realtime facial expression detector using Tensorflow and OpenCV (webcam input)

The project employs a basic tensorflow model trained on Google's inception model working on webcam input taken through OpenCV.


Code running on Python3

Dependencies: TensorFlow, OpenCV, NumPy


To run the program on your machine,

	1. Clone the repository
	
	2. Download database from https://drive.google.com/drive/folders/1hLdt1WGaBTWX9WI6Kgra_6diB2jj7ULF?usp=sharing into same directory as the repository
	
		2.1. (optional) create your own database by sorting images in CK Databse (mentioned below) and running cvtGray.py, face_crop.py, rename_to_jpeg.py
		
	3. Run retrain.py
	
	4. Run test.py
	

- retrain.py code originally by Google: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining	/retrain.py

Editted by caffeinatedfire


- haarcascade for face detection by Intel: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml


- Dataset was made using Cohn-Kanade AU-Coded Facial Expression Database: http://www.consortium.ri.cmu.edu/ckagree/
	- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE	  International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
	- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A 	complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR	 	for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
