#If code doesn't run make sure you've installed the necessary libraries
#python -m pip install -U scikit-image
#python -m pip install -U scipy
#python -m pip install -U matplotlib

#Directory containing your jp folders, folders should be named jp#, i.e jp3
jpd = r'/home/josh/Downloads/Brouard/julia_test_data/' 
fn = 'averaged_V+H_NO.dat'
jpt = 'socon' #is j' spin orbit conserving (socon) or changing (soch)

#What j' range are you looking at?
jpi = 3   #This is your initial j'
jpf = 3   #This is your final j'

#What are the velocities of Rg and NO, used to calculate ellipse angle
v_Kr = 531
v_NO = 617

#Only change this value if camera mount is changed
cr = 102

#Gaussian blurring strength (8 by default)
gfs = 8

#Ellipse intensity threshold (0.8 by default)
et = 0.85

#No need to look past this point unless updating code
import math
import os
import time
import sys 

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, rotate
from skimage import io
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

start = time.time()
debug = True

#Calculate ellipse angle
ellipse_angle = math.degrees(math.atan(v_Kr/v_NO)) + 90 - cr
#Calculate angle in radians
angle = np.radians(ellipse_angle)
cos, sin = np.cos(angle), np.sin(angle)

#For spin-orbit conserving (socon) j' 2 to 15
vpon_NO =  [0.0,592.818734,586.9941437, 579.4190111, 570.0232933, 558.7148348, 
          545.3742928, 529.8477677, 511.9359121, 491.3773917, 467.8227965, 
          440.7913959, 409.5946369, 373.1884948, 329.8515513]
#For spin-orbit changing (soch) j' 3 to 15
vpch_NO = [0.0,0.0,523.3017729,514.5455156,503.6373246,490.4339901,474.7444843,
         456.3129313,434.7913264,409.6936487,380.3133355,345.5606264,
         303.5971966,250.8310563,177.8979948]
if jpt == 'socon': vp_NO = vpon_NO
else: vp_NO = vpch_NO

def load_image(picture):
    '''Construct image from labview output file.'''
    #Load the .dat into an array [x,y,colour]
    da = np.loadtxt(picture)
    #Extract your x,y, and z values
    x=da[:,0]
    y=da[:,1]
    z=da[:,2]
    #Get the unique values of x and y
    xu=np.unique(x)
    yu=np.unique(y)
    #Make an array of arrays with size len(x) by len(y)
    #For example an array of 300 x arrays containing 300 y values
    xp,yp = np.meshgrid(xu,yu)
    #reshape the colour data into the appropriate shape to match x and y
    zp = z.reshape(len(xu),len(yu))
    zp = rotate(zp, cr, reshape=False)
    zp = gaussian_filter(zp, sigma=gfs)
    return xp,yp,zp

def pointInEllipse(x,y,xp,yp,w,h,ellipse_check):
    '''Calculate if points are contained within ellipse.'''
    #tests if a point[xp,yp] is within boundaries defined by the ellipse
    #of center[x,y] and tilted at an angle.
    #Example code from stackoverflow: https://tinyurl.com/pvbznbpw
    #subtract center from points
    xc = xp - x
    yc = yp - y

    xct = xc * cos - yc * sin
    yct = xc * sin + yc * cos

    rad_cc = (xct**2/(w/2.)**2) + (yct**2/(h/2.)**2)
    #If we are checking if max intensities in ellipse
    if ellipse_check:
        check = np.where(rad_cc <= 1.)[0]
        #If all points not included ellipse, parameters fail
        if len(check) != len(xp): return [False,rad_cc]
        return [True,rad_cc]
    #Remove values outside of ellipse
    else:
        in_ellipse = np.argwhere(rad_cc >= 1.)
        xe, ye = in_ellipse[:,[0]],in_ellipse[:,[1]]
        return xe, ye

def find_best_ellipse(xp,yp,zp):
    '''Using the image points find best ellipse'''

    #Find the maximum intensity point
    zm = np.max(zp)
    #Find any points within threshold value
    zh = np.argwhere(zp>(zm*et))
    #Get x,y values for high intensity points
    max_xe, max_ye = zh[:,[1]],zh[:,[0]]
    #Find any points within threshold value
    zl = np.argwhere(zp<(zm*et))
    #Get x,y values for high intensity points
    min_xe, min_ye = zl[:,[1]],zl[:,[0]]
    #best ellipse placeholder
    be = {'cx':100,'cy':100,'w':60,'h':80,'score':0,'pc':0}
    bes,pcs = [],[]
    for cx in range(130, 170, 5): #loop through x centers
        for cy in range(130, 170, 5): #loop through y centers
            for w in range(60, 120, 10): #widths
                for h in range(80, 140, 10): #heights
                    #check if all points contained in ellipse
                    if debug == True:
                        pc = pointInEllipse(cx,cy,max_xe,max_ye,w*2,h*2,True)
                        #if sum(pc[1]) > be['pc']*0.8:
                        bc = pointInEllipse(cx,cy,min_xe,min_ye,w*2,h*2,True)
                        score = sum(bc[1])-sum(pc[1])
                        if pc[0] == True and be['score'] < score:
                            be = {'cx':cx,'cy':cy,'w':w,'h':h,'score':score,'pc':sum(pc[1])}
                            print(be)
                            bes.append(be)
                            pcs.append(sum(pc[1]))
                    else:
                        pc = pointInEllipse(cx,cy,max_xe,max_ye,w*2,h*2,True)
                        if pc[0] == True and be['pc'] < sum(pc[1]):
                            be = {'cx':cx,'cy':cy,'w':w,'h':h,'score':0,'pc':sum(pc[1])}
                            print(be)
    if debug == True:
        print(pcs)
        mpc = max(pcs)
        midx = pcs.index(mpc)
        be = bes[midx]
    return be

#Make a directory to save data to
pd = os.path.join(jpd,'ellipse')
try: os.mkdir(pd)
except FileExistsError: pass

#Loop through jp values
start = time.time()
for i, j in enumerate(range(jpi,jpf+1)):
    plt.figure()
    ax = plt.gca()
    #Check that directory is valid
    jd = os.path.join(jpd,f'jp{j}')
    if os.path.isdir(jd) == False:
        print(f"No directory found for j' of {j}.")
        continue
    #Check that file is valid
    jd = os.path.join(jd,fn)
    if os.path.isfile(jd) == False:
        print(f"No file found for j' of {j}.")
        continue
    #Load data from file
    xp,yp,zp = load_image(jd)
    #Get maximum x and y values for centring ellipse
    cw,ch = np.max(xp),np.max(yp)
    #Calculate best ellipse
    be = find_best_ellipse(xp,yp,zp)
    be['cx'] -= cw
    be['cy'] -= ch
    #Remove points outside ellipse
    xe, ye = pointInEllipse(be['cx'], be['cy'], xp, yp, be['w']*2.2, be['h']*2.2,False)
    zp[xe,ye] = 0
    #Plot data
    plt.pcolormesh(xp,yp,zp,shading='auto')
    #Make an ellipse, subtract maximum values as data ranges from -150 to 150 not 0-300
    ell_patch = Ellipse((be['cx'], be['cy']), be['w']*2.0, be['h']*2.0,
        180-ellipse_angle,edgecolor='red', facecolor='none',zorder=15)
    #Plot an ellipse
    ax.add_patch(ell_patch)
    #Theoretical radius, multiple velocity of NO by velocity to pixel factor
    cr = vpon_NO[j] * (1 / 5.2)
    ell_patch = Ellipse((be['cx'], be['cy']), cr*2.0, cr*2.0,
        180-ellipse_angle,edgecolor='green', facecolor='none',zorder=15)
    #Plot an ellipse
    ax.add_patch(ell_patch)
      
    if debug == 69:
        plt.scatter(be['cx'], be['cy'],color='green')
        rec = plt.Rectangle((-20,-20),40,40,edgecolor='green', facecolor='none',zorder=15)
        ax.add_patch(rec)
    plt.title(f"jp{j}, sx:{round(be['w']/cr,2)}, sy:{round(be['h']/cr,2)}, cx:{be['cx']}, cy: {be['cy']}")
    plt.savefig(os.path.join(pd,f'jp{j}'))
print(f'Finished in {time.time()-start} seconds')