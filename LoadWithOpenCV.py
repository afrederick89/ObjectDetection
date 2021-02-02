import cv2
filepath = 'C:\\Users\\Ryan\\Desktop\\Fullerton Fall 2020\\Tuesday.Thursday - CPSC 481 - Artificial Intelligence\\Final\\'

tensorflowNet = cv2.dnn.readNetFromTensorflow(model=filepath + 'Models\\TF2Model_Frozen_Graph.pb')
img = cv2.imread(filepath + 'Screenshots\\1.png')
rows, cols, channels = img.shape

blob = cv2.dnn.blobFromImage(img, size=(400,400), swapRB=False, crop=True)

# Use the given image as input, which needs to be blob(s).
tensorflowNet.setInput(blob)

# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()

# Loop on the outputs
for detection in networkOutput:
    score = float(detection[2])

    if score > 0.2:
        left = detection[3] * cols

        top = detection[4] * rows

        right = detection[5] * cols

        bottom = detection[6] * rows


        # draw a red rectangle around detected objects
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

# Show the image with a rectangle surrounding the detected objects
cv2.imshow('Image', img)
cv2.waitKey()
cv2.destroyAllWindows()
