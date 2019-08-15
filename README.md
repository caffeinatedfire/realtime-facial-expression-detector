# realtime-facial-expression-detector
Realtime facial expression detector using Tensorflow and OpenCV (webcam input)

The project employs a basic tensorflow model trained on Google's inception model working on webcam input taken through OpenCV.

retrain.py code originally by Google: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
Editted by caffeinatedfire

Code running on Python3
Dependencies: TensorFlow, OpenCV, NumPy
To run the program on your machine,
	1. Clone the repository
	2. Run retrain.py
	3. Run test.py

Dataset was made using Cohn-Kanade AU-Coded Facial Expression Database
	- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.
