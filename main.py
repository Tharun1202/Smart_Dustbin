import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import tensorflow as tf
import tensorflow_hub as hub
# Define GPIO pins for the motor coils
coil_A_1_pin = 17
coil_A_2_pin = 27
coil_B_1_pin = 23
coil_B_2_pin = 24

# Set GPIO mode and setup pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(coil_A_1_pin, GPIO.OUT)
GPIO.setup(coil_A_2_pin, GPIO.OUT)
GPIO.setup(coil_B_1_pin, GPIO.OUT)
GPIO.setup(coil_B_2_pin, GPIO.OUT)
print(tf.__version__)
#from tensorflow_hub import KerasLayer
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
def process_image_with_cnn(image):
    # Define your model architecture
    print("Defining model...")
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    # Create a KerasLayer object from the URL
    pre_trained_layer =  hub.KerasLayer(model_url, output_shape=[1280], trainable=False)
    # Define the complete model architecture
    model = tf.keras.Sequential([
      pre_trained_layer,
      layers.Dropout(0.5),
      layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.001)),
      layers.Dense(3, activation='softmax', kernel_regularizer=l2(0.001))
    ])
    model(tf.zeros((1, 224, 224, 3)))
    # Load the weights of the model
    print("Loading weights...")
    model.load_weights('/home/pi/Downloads/waste_classifier_save_weights.h5')
    print("Weights loaded successfully.")

    # Preprocess the image
    image = cv2.resize(image, (224, 224))  # Resize the image to the size your model expects
    image = tf.keras.preprocessing.image.img_to_array(image)  # Convert the image to a numpy array
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model prediction
    image = image / 255.0  # Normalize pixel values if your model expects it

    # Predict the class of the image
    prediction = model.predict(image)
    print(prediction)
    classification_result = np.argmax(prediction)  # Assuming your model uses categorical output

    return classification_result

# Function to set the stepper motor coils for a step
def setStep(w1, w2, w3, w4):
    GPIO.output(coil_A_1_pin, w1)
    GPIO.output(coil_A_2_pin, w2)
    GPIO.output(coil_B_1_pin, w3)
    GPIO.output(coil_B_2_pin, w4)

# Define the stepper motor sequence
StepCount = 4
Seq = [[1,0,0,1],
       [0,1,0,1],
       [0,1,1,0],
       [1,0,1,0]]

StepDir = 1 # Set to 1 for clockwise, -1 for counterclockwise
StepsPerRevolution = 2000  

# Placeholder for classification result (0 for e-waste, 1 for biodegradable, 2 for non-biodegradable)
classification_result = -1  # Initialize to an invalid value
def move_clockwise(steps):
    for step in range(steps):
        for pin in range(0, 4):
            setStep(Seq[step % StepCount][0], Seq[step % StepCount][1], Seq[step % StepCount][2], Seq[step % StepCount][3])
            time.sleep(0.001)  # Wait before moving to the next step

#def move_counter_clockwise(steps):
  #  for step in range(len(Seq) - 1, -1, -1):
   #     for pin in range(0, 4):
    #        setStep(Seq[step % StepCount][0], Seq[step % StepCount][1], Seq[step % StepCount][2], Seq[step % StepCount][3])
     #       time.sleep(0.001)  

# Run the stepper motor
while classification_result == -1:
    # Capture an image from the camera
    camera = cv2.VideoCapture(0)#initially the value was 0 but i've changed it to 14
    _, image = camera.read()
    camera.release()

    # Process the image with the CNN model for waste classification
    classification_result = process_image_with_cnn(image)
####ORIGINAL_VAL FOR BIO IS 1
####CHANGED_VAL  FOR BIO IS 0
####ORIGINAL_VAL FOR EWASTE IS  0
####CHANGED_VAL FOR EWASTE IS 1
    # Calculate target position based on classification result
    if classification_result != -1:
        steps_per_degree = StepsPerRevolution / 360.0  # Calculate steps per degree

  # Target positions based on steps_per_degree
    if classification_result == 1:  # e-waste
        target_position = int(steps_per_degree * 330)  # Convert to integer for step count
        print("e-waste")
    elif classification_result == 0:  # biodegradable
        target_position = int(steps_per_degree * 120)  # Convert to integer for step count
        print("biodegradable")
    elif classification_result == 2:  # non-biodegradable
        target_position = int(steps_per_degree * 250)  # Convert to integer for step count
        print("non-biodegradable")

        # Move the stepper motor to the target position
    if classification_result != -1:
        move_clockwise(target_position)

    # Introduce a 15-second delay after reaching the target position
        print("Delaying for 15 seconds...")
        time.sleep(15)
        # Return to starting position
        print("moving")
        # Calculate steps for full revolution
        full_revolution_steps = StepsPerRevolution
    # Move back by difference between full revolution and target position
        move_clockwise(full_revolution_steps - target_position)
        print("end")
    
# Stop the stepper motor
GPIO.cleanup()
