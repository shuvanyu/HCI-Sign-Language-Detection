#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os
import time
import mediapipe as mp

collection_mode = False


# ### Defining constants throughout the model

# In[2]:


def definitions():
    width_cam, height_cam = 1000, 1980

    # Actions we will detect in this code
    actions = np.array(["Hello", "Thanks", "Yes"])

    # Collect 30 different videos for each action above
    num_sequence = 40

    # Videos with 30 frames in length
    sequence_len = 30
    
    # path for the exported image data (numpy arrays)
    path = os.path.join("Sign_Lang_Data")
    
    # Frames per second
    fps = 30
    
    return width_cam, height_cam, fps, actions, num_sequence, sequence_len, path


# ### Keypoints and Landmarks using mediapipe Holistic

# In[3]:


def instantiation():
    # Instantiating the holistic model
    mp_holistic = mp.solutions.holistic

    # Drawing utilities to draw the landmark lines on the video/image
    mp_drawing = mp.solutions.drawing_utils
    
    return mp_holistic, mp_drawing


# ### Mediapipe holistic model to make detections of keypoints/landmarks

# In[4]:


def detection(img, mp_model):
    mp_holistic, mp_drawing = instantiation()
    
    # Convert default opencv BGR frame to RGB frame for image detection
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Setting the image writeable status to False
    img.flags.writeable = False
    # Detection of the image using mediapipe
    results = mp_model.process(img)
    # Making the image status to writeable
    img.flags.writeable = True
    # Convert RGB frame to BGR frame for opencv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img, results


# In[5]:


def draw_landmarks(img, results):
    mp_holistic, mp_drawing = instantiation()
    
    mp_drawing.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,255,121), thickness=1, circle_radius=1)
                             )
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


# ### Extracting landmark coordinates

# In[6]:


def extract_keypoints(results):

    # This will return a flattened array of pose landmarks. 
    # The results.pose_landmarks.landmark is an array [x, y, z, visibility] of shape (33,4)
    # len(results.pose_landmarks.landmark) = 33
    pose = np.array([[res.x, res.y, res.z, res.visibility]                      for res in results.pose_landmarks.landmark]).flatten()                         if results.pose_landmarks                         else np.zeros(33*4)

    # This will return a flattened array of face landmarks. 
    # The results.face_landmarks.landmark is an array [x, y, z] of shape (468, 3)
    # len(results.face_landmarks.landmark) = 468
    face = np.array([[res.x, res.y, res.z]                      for res in results.face_landmarks.landmark]).flatten()                         if results.face_landmarks                         else np.zeros(468*3)

    # This will return a flattened array of left hand landmarks. 
    # The results.left_hand_landmarks.landmark is an array [x, y, z] of shape (21, 3)
    # len(results.left_hand_landmarks.landmark) = 21
    left_hand = np.array([[res.x, res.y, res.z]                      for res in results.left_hand_landmarks.landmark]).flatten()                         if results.left_hand_landmarks                         else np.zeros(21*3)

    # This will return a flattened array of right hand landmarks. 
    # The results.right_hand_landmarks.landmark is an array [x, y, z] of shape (21, 3)
    # len(results.right_hand_landmarks.landmark) = 21
    right_hand = np.array([[res.x, res.y, res.z]                      for res in results.right_hand_landmarks.landmark]).flatten()                         if results.right_hand_landmarks                         else np.zeros(21*3)
    
    # returns a concatenated array of 33*4 + 468*3 + 21*3 + 21*3 = 1662
    
    return np.concatenate([pose, left_hand, right_hand])


# ### Set Camera Angle

# In[7]:


width_cam, height_cam, fps, _, _, _, _ = definitions()
mp_holistic, mp_drawing = instantiation()

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)
cap.set(cv2.CAP_PROP_FPS, fps)

mp_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5)

while True:
    # Read current frame
    ret, img = cap.read()

    # Image detection
    img, results = detection(img, mp_model)
    #print(results)

    # Draw Landmarks
    draw_landmarks(img, results)

    cv2.putText(img, text='SET CAMERA ANGLE', org=(120,200),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, 
                color=(0,0,255), thickness=2, lineType=cv2.LINE_AA)

    # Show the feed on screen
    cv2.imshow('Image', img)

    # Break the loop using the key 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ### Setup Folders for training data collection

# In[8]:


def folder_setup():
    _, _, _, actions, num_sequence, sequence_len, path = definitions()

    # This will create 3 folders called "Hello, "Thanks, "I Love You"
    # Each folder will have 30 sub-folders created inside them 
    # Each of the above sub-folders will have 30 frames captured for a video sequence
    # Hello
        ## 0
        ## 1
        ## ...
        ## 29
    # Thanks
        ## 0
        ## 1
        ## ...
        ## 29
    # I Love You
        ##

    for action in actions:
        for sequence in range(num_sequence):
            try:
                os.makedirs(os.path.join(path, action, str(sequence)))
            except:
                pass


# ### Collect Keypoint Values for Training and Testing

# In[9]:


def data_collection():
    folder_setup()
    width_cam, height_cam, fps, actions, num_sequence, sequence_len, path = definitions()
    mp_holistic, mp_drawing = instantiation()
    
    cap = cv2.VideoCapture(0)
    cap.set(3, width_cam)
    cap.set(4, height_cam)
    cap.set(cv2.CAP_PROP_FPS, fps)

    mp_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
    
    # Loop through the 'actions' list
    for action in actions:
        # Loop through the number of videos (sequences) to be captured
        for sequence in range(num_sequence):
            # Loop through number of frames in a video (video length)
            for frame_num in range(sequence_len):

                # Read current frame
                ret, img = cap.read()

                # Image detection
                img, results = detection(img, mp_model)
                #print(results)

                # Draw Landmarks
                draw_landmarks(img, results)

                # Wait logic for collecting frames 
                if frame_num == 0:
                    cv2.putText(img, text='STARTING COLLECTION', org=(120,200),
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                                color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(img, 'Collecting Frames for {}, Video Number {}'.format(action, sequence),
                               (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('Image', img)
                    # Will take 1 sec pause between each video
                    cv2.waitKey(2000)
                else:
                    cv2.putText(img, 'Collecting Frames for {}, Video Number {}'.format(action, sequence),
                                (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0,0,255), 1, cv2.LINE_AA)
                    cv2.imshow('Image', img)


                # Export keypoints block
                keypoints = extract_keypoints(results)
                # To specify the path to save the numpy array of the frame
                npy_path = os.path.join(path, action, str(sequence), str(frame_num))
                # To save the numpy array of the frame
                np.save(npy_path, keypoints)

                # Break the loop using the key 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
if collection_mode:
    data_collection()

