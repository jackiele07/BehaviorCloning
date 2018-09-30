
# **Behavioral Cloning**


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./write_up_image/VGG1_3epoch_wo_dropout.png "3 epoch without drop out"
[image2]: ./write_up_image/VGG1_3epoch_dropout.png "3 epoch with drop out"
[image3]: ./write_up_image/VGG2_5epoch_dropout.png "5 epoch with drop out"
[image4]: ./write_up_image/VGG2_5epoch_w_flipped_dropout.png "5 epoch with dropout - more data"
[image5]: ./write_up_image/VGG2_3epoch_w_flipped_dropout20.png "3 epoch with less dropout - more data"
[image6]: ./write_up_image/VGG_model.png "model architecture"



### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* VGG_model.h5 containing a trained convolution neural network
* Report to summarize the result
* video of result as run1.mp4

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py VGG_model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

There are 2 model I am using for this project. First is an dirty model, which is used during the lecture for quickly ramp up with the project and get the direction for tunning model.

This dirty model includes:
- 1 Lambda layer to normalize the image pixel
- 1 Cropping2D to eliminate the non-senses information in the image, which are skies, trees, and so on
- 2 Conv2D followed by MaxPolling2D and Dropout by 25%
- 2 Fully connected layer

Later, it turned out that the dirty model is too simple to be tunned, and the error is still quite far from expectation. Then I decided to use more complicated model, the most easy and handy model seems to be VGG since I have already experimented it earlier during my previous project.

The VGG-like network includes:
- 1 Lambda and 1 Cropping2D for preprocessing before entering to network
- 3 convolution block which each block consist of 2 Conv2D followed by MaxPooling2D and Dropout by 20%
- 4 fully connected layer

Both the layer using adam optimizer and mean squared error to compile the model. I do not have enough time to change to another loss function than mse, but since this is linear regression problem, mse seems to be fit for use. Furthermore, I do not think that it will affect much to our final output of model in this case.

#### 2. Attempts to reduce overfitting in the model

Well, I have tried several attempts to reduce overfitting in each of the models.
- Ultilize the dropout and keep changing the %number to find the most optimum value
- But because of using dropout, it turned out that the model is hardly to be converged, so that I increase the train_samples by adding left & right image from IMG data with applying some correction factor. Later, I also applied the flip image to train_samples as well to increase the train_samples totally by 5 times
#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25). batch_size is calculated due to my local graphics card memories, I have 4.96GiB freememory, so the best batch_size is arround 6 - 14. I select 6 as I do not want to use all of my GPU Ram for training :), as shown in task manager, 5.4/6.0GB is used, that is extremely fine for me.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road as well as flipped left sides and right side images to increase the training data by 5 times

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first get my hand dirty by quickly implement a dirty model an run the first trial with all the tool so that I could get familiar with the project as well as getting understand all the constraints of the problem in this project.

By quickly analyze the dirty model, I could see that only using the center image will causing the problem because there is a certain error of the model's output which could lead to the car position is lying on the most left or most right of the road, where it cannot be recover to the center point because there is not such kind of data in training sample. (well, I did this project in a rush after my business trip so that I did not realize that David already mentioned about this issue in lecture video <~_~> )

After working with dirty model, I have everything set very quickly, I could try several run with simulator and see what need to be done and also tried to tune some model parameter (dropping factor, epoch number, increase training size...) to have better result. But it turned out that I need more complicated model for this problem to achieve accurate steering angle output, because the dirty model is easily overfitting after only 1 epoch and the loss is quite high

I decided to use VGG-like model since it is very easily implemented and handy. Furthermore, I have also check out the solution from NVIDIA team as recommended, I could see it also VGG-like model. So it convinced me that a VGG-like model is well fit for this problem

To combat the overfitting, I modified the model many times to add more dropout layer, find the most suitable dropout factor and also add more linear layer to the model and increase the epoch number.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. It is so strange for me since all of the number in model showing me nothing wrong with the model. Then I think there is something wrong not with my model, I spend time to go through all of the note in lession carefully and figure out the color space mismatch between my model and drive.py :( The ultility is design for RGB color space while my model is getting the input from BGR. Then I have tried again and finally the result looks pretty good (^_^)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

As described earlier, the final model of mine is VGG-like model which includes:
- 1 Lambda and 1 Cropping2D for preprocessing before entering to network
- 3 convolution block which each block consist of 2 Conv2D followed by MaxPooling2D and Dropout by 20%
- 4 fully connected layer

Here is a visualization of the architecture which is generated by using ```keras_plot```

![VGG-like model][image6]

#### 3. Creation of the Training Set & Training Process

I have experieced to collecting the data by myself as well, but it turned out that I must focus more on the model first until the sample data does not satisfy me... My plan to collect the data by myself for the 2nd track and train my model to see if it still fit to solve the problem or not. But for submission (since it is already overdue) I only includes my work for the first track.

I have made several plot of each model to see how I could improve my model.

First I have tried to run the model with 3 epoch, using the all 3 camera images, I got the output as following

![3 epoch full data][image1]

Easily, my model got overfitted after first epoch. It seems that I need to apply regularization or dropout. I try dropout after each MaxPooling2D layer (3 times in whole model) with factor of 0.2 (20%), and run with 3 epoch, I got the result as following:

![3 epoch fulldata w dropout][image2]

It is very much better, and it seems to be able to reduce even more with more epoch. Then I decided to increase the epoch number to 5 with the hope that I could get the loss of validation set around 0.012. The result is as following:

![5 epoch full data w dropout][image3]

Well, it turned out that I get the better result, but the fig look strange to me since the model look overfitting after 3 epoch and then it recovered itself somehow. Maybe it is good idea to increase my dropout factor a little bit, but if I do so, the model would not able to converged according to the chart. Then I decided to increase my training data insteads, I flipped all the left & right image and applied correction to steering data by multiply with -1.0. By that way I increased the training data by 60%. I made another run with exsiting model and observed the output:

![5 epoch agumented data w dropout][image4]

Extremly better!!! My model works very well and the error reduce to ~ 0.012 which meets my expectation already (^_^) After this, I have also tried to create more complicated model like adding more convolution layer and fully connected layer. But unfortunately, the result is worse compare to this model. It seems I need to collect more data if I want better result, in future I will try another network like ResNet or GoogLeNet and see how it perform, but for now, I think it is sufficient to submit for this project and I am happy with this result already :)
