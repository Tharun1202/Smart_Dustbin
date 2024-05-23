import RPi.GPIO as GPIO
import time
import subprocess

# Define GPIO pins
TRIG_PIN = 22  # 22
ECHO_PIN = 25  #25
# old 23 24
# Set up GPIO pins
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
SPEED_OF_SOUND = 34300

def measure_distance():
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.000002)
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)

    # Wait for echo pulse to go high
    while GPIO.input(ECHO_PIN) == 0:
        pass
    pulse_start = time.time()

  # Wait for echo pulse to go low
    while GPIO.input(ECHO_PIN) == 1:
        pass
    pulse_end = time.time()
    # Measure echo pulse duration
    pulse_duration = pulse_end - pulse_start

  # Calculate distance
    distance = pulse_duration * SPEED_OF_SOUND / 2

    return distance


# Main loop
try:
    while True:
        distance = measure_distance()

        # Check if object is within range
        if 10 <= distance <= 15:
            # Trigger another Python program using virtual environment path
            print("Motion detected at", distance, "cm")
            subprocess.run(['/home/pi/myenv/bin/python', '/home/pi/main.py'])
            print("Motion detected at", distance, "cm")
            # Optionally, add a delay to avoid triggering too frequently
            time.sleep(1)
        else:
            print("Motion detected at", distance, "cm")
            break
        time.sleep(0.1)  # Adjust delay as needed

except KeyboardInterrupt:
    # Clean up GPIO on exit
    GPIO.cleanup()
    print("Exiting program")
