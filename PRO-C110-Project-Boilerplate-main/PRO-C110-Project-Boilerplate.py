# To Capture Frame
import cv2

# To process image array
import numpy as np
import tensorflow as tf

# import the tensorflow modules and load the model
model=tf.keras.models.load_model("kares_model.h5")



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = camera.read()

	# if we were sucessfully able to read the frame
	if status:

		# Flip the frame
		frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
			
		img=cv2.resize(frame,(224,224))
		testimg=np.array(img,dtype=np.float32)
		testimg=np.expand_dims(testimg,axis=0)
		image=testimg/255.0
		prediction=model.predict(image)
		print(prediction)

		# expand the dimensions
		
		# normalize it before feeding to the model
		
		# get predictions from the model
		
		
		
		# displaying the frames captured
		cv2.imshow('feed' , frame)

		# waiting for 1ms
		code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
		if code == 32:
			break

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()
