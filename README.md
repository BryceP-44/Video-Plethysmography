# Video-Plethysmography
Reads heartrate from a video of a human face. 
Run the eye tracker program first and then run the pleth program. Put the video file in the same folder as the code. Watch the output video as the pleth code is running to see if you need to change "xstretch" and "raisefactor" based on the individual's facial shape. It is still very experimental and not finished. Heartrate can definetely be seen in the output though. I think that we tend to bob our heads when a blood pulse creates a torque in our neck and that is what is being seen... So, maybe looking at the middle of both eyes' position over time would show our eyes making slight jerking movements every blood pulse. I don't code for UPMC anymore, so I don't really care enough. Maybe I'll make a video one day on this topic though. 
<br />
Here is an example of a new pytorch weights file that I am using to detect myself. I put links to 2 videos in the eyetracker code if you want to train your own.
<br />

![detectbryce](https://github.com/BryceP-44/Video-Plethysmography/blob/main/detectbrycepic.PNG)
