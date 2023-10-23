# Muslim-prayer-recognition-mobile-application

This project analyzes whether a Muslim's prayer is correct or not.
There are five prayers for Muslims that differ in the number of rak'ahs.
The dawn prayer is two rak'ahs, the noon, afternoon, and evening prayers are four rak'ahs, and the sunset prayer is three rak'ahs.
The movement is analyzed from the accelerometer sensor signal on the three axes, and the movement is described as whether it is standing, kneeling, sitting, or prostrating through a classifier based on deep learning (CNN model).
The project consists of two aspects: server and mobile application.
The mobile application sends the accelerometer sensor signal to the server.
The server classifies prayer movements based on the accelerometer sensor data, from which it determines the accuracy of the prayer and sends the result to the application.
