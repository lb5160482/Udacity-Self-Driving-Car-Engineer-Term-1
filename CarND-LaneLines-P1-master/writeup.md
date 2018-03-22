# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps: 
* First, I converted the images to grayscale.

* Then I use Gaussian blur to smooth the image with a filter size 5. 

* Next, I use Canny edge detector to extract edges in the image using min and max threshold to be 50 and 150. 

* After that, I set the ROI with four points that only enclose the two lanes. 

* Then I use Hough Transform to get and draw the straight lines. I tuned the parameters for a while and finally use rho as 1, theta as 1 degree, minimum number of votes as 10, minimum number of pixels making up a line as 15 and maximum gap in pixels between connectable line segments as 1. 

* Finally I overlaid the lines on the original image. 


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by:  
* Checking the length of the line by 
```python
if math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < 40:
	continue
```
If the line is too short, then I wil ignore. This is like removing the noise. 

* And then I computed the slopr of each line and check if it is a left side lane or a right side lane. Then extent them correspondingly.
```python
slope = (y1 - y2) / (x1 - x2)
if 0.5 < abs(slope) < 10:
    if slope < 0: # /
        x2 = int(img.shape[1] / 2)
        y2 = int(y1 + (x2 - x1) * slope)
        y1 = int(img.shape[0])
        x1 = int(x2 - (y2 - y1) / slope)
    else: # \
        x1 = int(img.shape[1] / 2)
        y1 = int(y2 + (x1 - x2) * slope)
        y2 = int(img.shape[0])
        x2 = int(x1 - (y1 - y2) / slope)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
```


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be in some specific frame the the lines and the broken lane side might not be detected.

Another shortcoming could be based on the noise on the road, some straight lines that are not road lanes might also be detected.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to keep tuning the threshold so that there will be no line missing.

Another potential improvement could be to check the majority slope of those lines we detected so that we can eliminate the straight lines.
