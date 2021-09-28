#Bryce Patterson Video Plethysmography Beta Version 1.0

from math import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cv2
import numpy as np
import time
from scipy import signal
from scipy.fft import fft, ifft,fftfreq
from scipy.signal import butter, sosfilt, sosfreqz,freqz
import csv
import pandas as pd

"""change these lines manually"""
#these 4 values must be changed manually for every video
smooth=20 #20 is a nice sweet spot number
videosmooth=20 #this is to smooth out the coordinates for 
xstretch=.7 #this is the horizontal ratio factor
raisefactor=0.5 #this is the vertical ratio factor
cap = cv2.VideoCapture("short3.mp4") #make the video the same one used in eyetracker



frames = int(cap.get(7)) #total frames
print(frames," frames") #display # of frames
width=cap.get(3)
height=cap.get(4)
print("Video Resolution: ",round(width)," x ",round(height))

f = open("colors.csv", "w")
f.truncate()
f.close()

csvout=['blue','green','red']
with open('colors.csv','a',newline='') as file:
    writer= csv.writer(file)
    writer.writerow(csvout)



#define the eyes and forehead function
def eyes(frame,outvid1,outvid2,width,height,xstretch,raisefactor,count,rstay,gstay,bstay,info,lag,framenum,df):
    ti=time.time()
    total=0
    out=[0,0,0,0,0,0,0,0,0,0,0,0,0] #Potential-x, x,Dx,y,Dy, Potential-y, used(1 or 0), avgb, avgg, avgr
    x=df['xmid'][framenum]
    y=df['ymid'][framenum]
    dx=df['Dx'][framenum]
    dy=df['Dy'][framenum]
    yolowidth=df['Width'][0]
    yoloheight=df['Height'][0]

    if x==0 and y==0 and dx==0 and dy==0 and framenum>0:
        x=df['xmid'][framenum-1]
        y=df['ymid'][framenum-1]
        dx=df['Dx'][framenum-1]
        dy=df['Dy'][framenum-1]


    x=x*width/yoloheight
    y=y*height/yolowidth
    dx=dx*width/yoloheight
    

    out[0]=x
    out[5]=y
    out[1]=x
    out[2]=dx
    out[3]=y
    
    keep1=out[1]
    keep2=out[2]
    keep3=out[3]

    
    #these values are defined at the top of program
    #this is to make the forehead box from the average position and x-displacement
    #remember that images have reversed y-axis
    x1=round(out[1]-xstretch*out[2]) #x position and go left a multiple of the x-displacment
    y1=round(out[3]-1.9*raisefactor*out[2])  #y position and go left a multiple of the x-displacment
    x2=round(out[1]+xstretch*out[2]) #same as x1 but other direction
    y2=round(out[3]- 1.1*raisefactor*out[2]) #same as y1 but other direction


    #draw some shapes to see where the forehead box and middle of eyes are
    """"""
    cv2.circle(img=frame, center=(int(out[1]), int(out[3])), radius=5, color=(255, 255, 0), thickness=-1)
    cv2.line(frame, (round(out[1]-out[2]), round(out[3])), (round(out[1]+out[2]), round(out[3])), (255, 0, 255),thickness=3)
    cv2.rectangle(frame, (x1, y1), (x2,y2), (255,255,255), 1)
    #cv2.rectangle(img, (x1, y1), (x2,y2), (255,255,255), 1)
    

    #minibox dimensions
    #the minibox is a box that sits at the bottom of the forehead box
    #it is there to make the forehead data much better
    #the average of the minibox is essentially meant to never have any hair in it
    #later, the minibox average can be used to get rid of any hair,moles or shiny parts on the forehead
    x21=round(x1+(x2-x1)*.25) #x with subscripts 2 1
    x22=round(x1+(x2-x1)*.75) #the box sits 50 less wide than original
    y21=round(y2-(y2-y1)*.40) #the box also sits only from 5 to 40 percent up from bottom of original box
    y22=round(y2-(y2-y1)*.05)
    #cv2.rectangle(frame, (x21, y22), (x22,y21), (255,255,255), 1)
    
    
    
    #initiate some variables
    avgb=bstay
    avgg=gstay
    avgr=rstay
    #img=frame
    n=0
    ################################################
    ###############################################
    ###############################################
    #get average of minibox
    
    avg=np.average(np.average(frame[y21:y22+1,x21:x22+1,:],axis=0),axis=0)
    avgb,avgg,avgr=avg
    
    mb=avgb #mb stands for mini blue
    mg=avgg
    mr=avgr
    out[10]=mr
    out[11]=mg
    out[12]=mb

    #ti=time.time()
    b=frame[y1:y2+1,x1:x2+1,0]
    b[.5*mb<b]
    b[b<1.8*mb]
    avgb=np.average(b)

    g=frame[y1:y2+1,x1:x2+1,1]
    g[.5*mg<g]
    g[g<1.8*mg]
    avgg=np.average(g)

    r=frame[y1:y2+1,x1:x2+1,2]
    r[.5*mr<r]
    r[b<1.8*mr]
    avgr=np.average(r)

    img=frame
    img[0:y1,:,:]=0
    img[y2+1:height,:,:]=0
    img[:,0:x1,:]=0
    img[:,x2+1:width,:]=0
    #write the whole frame with forehead box to video
    #outvid1.write(frame)
    outvid2.write(img)

    
    
                
    out[7]=avgb #these are the actual colors being sent out of this function every frame
    out[8]=avgg
    out[9]=avgr
    out[1]=keep1
    out[2]=keep2
    out[3]=keep3
    return out



frames = int(cap.get(7)) #frames
width= int(cap.get(3)) #width
height = int(cap.get(4)) #height
fps = cap.get(cv2.CAP_PROP_FPS) #capture fps
fourcc = cv2.VideoWriter_fourcc(*'MJPG') #needed to write the video out
outvid1 = cv2.VideoWriter('output1.avi', fourcc, fps, (width, height), isColor=True) #these are the output videos
outvid2 = cv2.VideoWriter('forehead.avi', fourcc, fps, (width, height), isColor=True)

times=[]
t=0
diff=1/30
secs=frames*diff
ti=time.time()
count=0

#this section is supposed to give a test video to see if the xstretch and raisefactor are good
#those values move the forehead box to its position but every video is different
#it outputs as black and white and I never got to fix it
#its stupid that it requires manual positioning, but just go watch "output1.avi" as its making it
#rerun the program and adjust those 2 values right under the libraries

"""
go="n"
while go=="n":
    xstretch=float(input("xstretch (1.5 is normal): "))
    raisefactor=float(input("raisefactor (1.2 is normal): "))
    for i in range(40):
        
        ret, frame = cap.read()
        length,width,channels = frame.shape
        nums=eyes(frame,testvid1,testvid2,width,height,cas1,cas2,cas3,cas4,cas5,cas6,cas7,cas8,cas9,xstretch,raisefactor)
        times.append(t)
        t+=diff

    cap2=cv2.VideoCapture('testvid2.avi')
    while(cap.isOpened()):
        ret, frame = cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    #clip = VideoFileClip('testvid1.avi')
    #clip.preview()
    go=input("Was the xstretch and raisefactor good? (y/n): ")

"""
###
lag=50
###


print("Math time...")
greens=[]
blues=[]
reds=[]
ti2=time.time()
counter=0
rstay=0
gstay=0
bstay=0
info=[]
df=pd.read_csv('coords.csv')

for i in range(lag*3+1):
    info.append(0)
for i in range(frames-1):
    #print(counter)
    counter+=1
    if count==0: #this is all for the fun little countdown timer, dont worry about it
        ti2=time.time()
        percenti=round(i*100/frames,2)
    if count==90:
        percentf=i*100/frames
        pleft=100-percentf
        deltap=percentf-percenti
        deltat=time.time()-ti2
        ti2=time.time()
        tleft=(deltat/deltap)*pleft
        mins=floor(tleft/60)
        if mins==1:
            secs=tleft%60
    
            #print(round(100*i/frames),"%:",i,"/",frames," frames done with ",mins," minute and ",round(secs)," seconds left")
            print(mins," minute and ",round(secs)," seconds left")
        if mins>1:
            secs=tleft%60
            #print(round(100*i/frames),"%:",i,"/",frames," frames done with ",mins," minutes and ",round(secs)," seconds left")
            print(mins," minutes and ",round(secs)," seconds left")
        if mins==0:
            secs=tleft%60
            #print(round(100*i/frames),"%:",i,"/",frames," frames done with ",round(secs), " seconds left")
            print(round(secs), " seconds left")
        count=0
        tf2=time.time()
        percenti=percentf
    count+=1
    ########################################
    #this is the slowest line of code in the whole thing
    ret, frame = cap.read() #read the video frame
    
    length,width,channels = frame.shape
    
    #this is the line that finally finally calls that forehead finder function
    nums=eyes(frame,outvid1,outvid2,width,height,xstretch,raisefactor,counter,rstay,gstay,bstay,info,lag,i,df)
    
    
    rstay=nums[10]
    gstay=nums[11]
    bstay=nums[12]
    if nums[7]!=0:
        blues.append(nums[7]) #if the function popped out a blue value, add it to an array called "blues"
    if nums[8]!=0:
        greens.append(nums[8]) #same thing but with green
    if nums[7]!=0:
        reds.append(nums[9]) #same thing but with red
        
    times.append(t) #honestly I dont remember what this is even used for but I'm pretty sure it has
    t+=diff #to do with one of the video functions from one of the libraries
    if counter==60:
        counter=0
    
    
cap.release() #release the video file
outvid1.release() #release the thing



#####################################################
#####################################################
#####################################################
#Completely done with the video processing and onto the color to heart rate conversion
#right now its just doing the reds

"""
print(" ")
print(reds) #uncomment and you can paste it into test17 at the top if you want
print(" ") #then you dont need to process a video every time and just run the color to HR part of the program
print(blues)
print(" ")
print(greens)"""

csvout=[blues,greens,reds]
with open('colors.csv','a',newline='') as file:
    writer= csv.writer(file)
    writer.writerow(csvout)


fig=plt.figure(figsize=(24, 12)) #add subplots
ax1=fig.add_subplot(211)
ax4=fig.add_subplot(212)

def make(reds,color):
        def smoother2(smooth, reds):
            for i in range(smooth):
                mult=1+.5/(i+300000)
                for j in range(len(reds)-1):
                    beforer=reds[j-1]
                    afterr=reds[j+1]
                    partavgr=(beforer+afterr)/2  
                    if reds[j]>mult*partavgr:
                        reds[j]=partavgr+(reds[j]-partavgr)*.5
                    if reds[j]<partavgr-(mult*partavgr-partavgr):
                        reds[j]=partavgr-(partavgr-reds[j])*.5
            out=reds
            return out

        def smoother1(smooth, reds):
            total=0
            for i in range(len(reds)):
                total+=reds[i]
            avgr=total/len(reds)
            for i in range(smooth):
                mult=1+.5/(i+.1)
                for j in range(len(reds)-1):
                    beforer=reds[j-1]
                    afterr=reds[j+1]
                    partavgr=(beforer+afterr)/2
                    
                    if reds[j]>mult*avgr:
                        reds[j]=avgr+(reds[j]-avgr)*.5
                    if reds[j]<avgr-(mult*avgr-avgr):
                        reds[j]=avgr-(avgr-reds[j])*.5
            out=reds
            return out

          
        ax1.set_ylabel("Red video")
        ax1.plot(reds,color)
        ykeep=reds
        
        reds=smoother2(20,reds)
        ax1.plot(reds,color)

        y=reds
        extremx=[]
        for i in range(1,len(y)-1): #this is to see where the peaks are in its x-coord
            a=y[i+1]
            b=y[i-1]
            slope1=y[i]-b
            slope2=a-y[i]
            if slope1>0 and slope2<0:
                extremx.append(i)

        highsred=[]
        for i in range(len(extremx)): #gets the y-coord of a peak from the x-coord
            highsred.append(reds[extremx[i]])
            
        ax1.plot(extremx,highsred,"g*")


        extremy=[]
        for i in range(len(extremx)):
            extremy.append(y[extremx[i]])
        print(len(extremy))
        instaheart=[]
        for i in range(len(extremx)-1):
                delta=extremx[i+1]-extremx[i]
                instaheart.append(1800/delta)
        instaheart.append(1800/delta)


        total=0
        instaheart=smoother1(4,instaheart)
        instaheart=smoother2(3,instaheart)
        xp=signal.resample(instaheart,100) #this is because they dont all find the same number of peaks
        #I just resample the values to 100 so all the colors get sync'd together
        for i in range(len(instaheart)):
            total+=instaheart[i]
        print("Average instaheart is ", total/len(instaheart)," BPM")

        ax4.plot(xp,color)
        ax4.set_ylabel("Heartrate")
        out=xp
        return out

red=make(reds,"r")
green=make(greens,"g")
blue=make(blues,"b")

HR=[]
for i in range(100): #get an average heartrate from all 3 colors
        HR.append((red[i]+green[i]+blue[i])/3)

#a=-.002272727
#b=.507818182
#c=-37.675
#d=929.4314547
a=.0019625512
b=-.293768286
c=12.11485898
for i in range(len(HR)):
    x=HR[i]
    mult=a*x**2+b*x+c #a*x**3+b*x**2+c*x+d
    HR[i]*=mult

ax4.plot(HR,"k.-")

plt.grid()
plt.show()
    


























