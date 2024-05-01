NeuralQ Specification
Overview
pass
Features
pass
Installation
Client install
Download neuralc package. 
Run following command to install dependencies,
pip install -r requirements.txt

Then simply run,
python NeuralC.py

Server install
Download neuralq package. Before starting, please install nvidia-docker2 is installed on server.
Then simply run,
docker-compose up -d

User's Guide
Sign-up/Sign-in
If you are using NeuralQ client for the first time, please switch to the server tab to configure server settings. Click connect to test connection. ( Server port is 2020 by default)
 
After connection established, an online will show. Otherwise, please check whether IP/port or the NeuralQ server is working correctly. 
For new user to the NeuralQ server, please sign up before using the system.
 
Now you are ready to sign-in!
Navigation
Before engaging in your project, there is a one last step to take. All samples are read from Neo4j database. Thus, please make sure the video file are stored in server db-directory and have been put into database. A quick way to do that is to click: Tools->Server->Connect

Choose a video file or a directory which contains multiple video file and click Choose button on botten right. Please Note: If you select a directory, only the video in its subdirectory are added (sub-subdirectory are not added!)
Getting start with a project
We are ready to go! Press: file->Create Project,
 
Then click Select button to add a video. Currently, a single project allows only one video only. After the wait dialog hide automaticly, double click the project you just created to browse the video！（First load may take some time.）

Available function
As shown on the control panel. Two functions are available at the moment, Tracing and Segmentation.
Tracing: 
pass (an short introduction to tracing here)
Segmentation:
pass (an short introduction to segmentation here)
Simple example
Let's take a few minutes to review the whole procedure.
Tracing
first select the rectangle on Marker panel to make a ROI zone which contains the object to track. The size of the rectangle area should be as small as possible. Please make sure the rectangle item is on the first frame of the video and there exists only one rectangle.

switch to Tracing tab. Press Send ROI Zone. Wait until the task finishes
 
Press Mark ROI Zone, and click play to browse the result. A bbox annotation should be applied to every frames now.
Switch to Marker tab again. Select the circle to mark the item to track. Here, multiple neurons are marked in order. ID of the circle label will show by mouse hovering. To edit the labels, use an eraser or switch to the cursor to move the item around. 
Some key poings: 
1. Make sure the number of circle labels are equal and consistent in each frame
2. All circle labels should be within the bbox of the current frame
3. Mark 5-20% number of total frames to ensure accuracy (minium 10 frames, 2 circle labels per frames)
4. Marked frame should be evenly distributed. (e.g. to mark 10/100 frame, mark the 4th, 14th, 24th... frame)
5. To collect high-accuracy annotation, use dotted rectangle on Marker to zoom in and double right click to zoom out
 

After the labouring work, hand everything to NeuralQ by clicking Send Annotations on the Tracing tab! A tracing task should begin now
 
When the task finishes, Press Mark Points to automaticly display all results. Make some corrction by draging the wrong-labelled item until satisfied with the result. Press Save Result to generate a .csv file in the workspace which records the coordinate of every single labels.

What's more, try out Analyze Result to generate a gray scale curve as well as a .csv file to see the change of gray scale value
Segmentation
Switch to Segmentaiton tab, choose the right muscle type, then Start Segmentation.

When segmentation finishes, press Plot Segmentation to see the result! Each segmentation represents a piece of individual muscle 

