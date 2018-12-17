# City Prediction [start](#0)
### Wentao Li
### unlimitediw@gwmail.gwu.edu

**************
<a name = "0"></a>
### Abstract [next section](#1)
***LosAngeles,United States vs Eindhove,Netherlands***
> Map Compare  

* Road Map  
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/LosAngelesRoadMap.png)  ![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/EindhovenRoadMap.png)

* Satellite Map   
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/LosAngelesSatellite.png)  ![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/EindhovenSatellite.png) 

> Feature Compare
* Los Angeles only has 144,140 population. What is the problem?
- I use python regex to match and combine cites features from different files. There are two cities named with Los Angeles, One is in United States with 3,884,307 population and the other one is in Chile. To remove these wrong data file, I select use the double check with both cities name and countries at the data preprocessing part.

* Los Angeles and Eindhoven City Features Comparison  
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/LosEnCityFeatures.png)

* GDP (label) Comparison   
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/LosEnGDP.png)

> What I want to do

* City Features to GDP  
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/FeaturesToGDP.png)

* Map to Population  
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/MapToPopulation.png)

* Background to RoadMap  
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/BackToMap.png)

* Background to Satellite  
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/BackToSatellite.png)

*****************************
## Part 1: City Data Collecting and Preprocessing
* In this project, all data is raw, collected and preprocessed by myself. 
> Numeric Data:
* The 4054 cities population and coordinate data is from [ergebnis](https://fingolas.carto.com/tables/ergebnis/public) publiced in 2014
* The 313 cities GDP and corresponding feature (Georgraphic and administrative forms) data is from [OECD](https://stats.oecd.org/Index.aspx?QueryId=51329#) 
* There are also many [other important features](https://github.com/unlimitediw/MLFinalProject/blob/master/DataRef.md) may help me to get higher accuracy in city GDP prediction. However, my focus points should be the model construction and optimization and is not familiar to deal with the deep city statistic such as "population by age" or "labour markert".
* My Data Preprocessing for combining these two .cvs file is [datapreprocessing](https://github.com/unlimitediw/DataSearch-Preprocessing/blob/master/DataPreprocessing.py)
* The data preprocessing includes:
  1. Removing Nan value.
  2. Mergeing city feature with city gdp, population and coordinate with city name regex and finally get 229 cities data.
  3. Changing all feature data type into numpy.float64.
  4. Save the new cities data in .csv format for future usage.
  5. When applying this data, I use standard normalization to all features except coordinate feature and number counting feature.
#
      def stdScl(F):
        F = (F - np.average(F))/np.std(F)
        return F
      def nor(F,scale):
        return F/scale
        
> Map Data:
* With the name of 4054 cities (collected in the numeric part), I insert it into my google map API. The format is https://maps.googleapis.com/maps/api/staticmap?&center=DC,USA&zoom=10&format=png&maptype=roadmap&style=element:labels%7Cvisibility:off&size=640x640&scale=2&key=AIzaSyCBj1rX0X4g7KaGueU1du_l4jzGfIQO1NY  
[Road](https://maps.googleapis.com/maps/api/staticmap?&center=DC,USA&zoom=10&format=png&maptype=roadmap&style=visibility:off&style=feature:road|visibility:on&style=element:labels%7Cvisibility:off&size=640x640&scale=2&key=AIzaSyCBj1rX0X4g7KaGueU1du_l4jzGfIQO1NY)   
[Satellite](https://maps.googleapis.com/maps/api/staticmap?&center=DC,USA&zoom=10&format=png&maptype=satellite&style=element:labels%7Cvisibility:off&size=640x640&scale=2&key=AIzaSyCBj1rX0X4g7KaGueU1du_l4jzGfIQO1NY)  
* The city name can also be replaced by coordinate likes (54.321,-12.345).
* With this url, you can easily adjust the format of map image you want. For instance, you can choose maptype = "roadmap" or "satellite" and adjust the zoom to get the scale you want for the city map.
* In my project, I coolect four kind of map data: roadmap, roadmap without road, roadmap withou map and the satellite map.
* I also used 60 threads to collect the data to improve speed.
* The collecting tool is [my Search.py](https://github.com/unlimitediw/DataSearch-Preprocessing/blob/master/Search.py)
## Part 2: Support Vector Regression
* I implement my own [SMO function](https://github.com/unlimitediw/MLGWU/blob/master/ML/CS6364_HW3_SVM_Handwork.py) for weight tunning in this part. But still use the sklearn.svm to train my model for higher speed.
* [SVR Reference](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf)
* After multiple C trying with 10-cross validation, I select C = 90,000. [my own k-fold api](https://github.com/unlimitediw/MLGWU/blob/master/ML/KFoldValidation.py)
* rbf is basically better than poly
* Larger training set can avoid overfitting
* The score of it is about 0.43 for 10-cross validation. And the score function is below:
#
      score = abs(k[i] - yTest[i])/yTest[i])
      # I finally decided to use yTest rather than max(yTest,hypo) because it can better fit my lost function. The hypo can be small but shouldn't be too large. If use max(yTest,hypo) the validation error for both svm and MLP will be about 0.3 and will not change.
* Learning Curve with different C value: (Y unit is 'abs(hypo - y) / y', just wrong writing)
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/SVMLcurveCValue.png)
* Learning Curve with different training data size (this curve is with seed = 3, I try many seeds, this is a bad one. However, at least it is a random choose point. The reason that 80% data validation error is larger than 60% one is that my dataset is too small (229 cities). There is a large flictuation for different random setting):
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/SVMLcurveSize.png)



## Part 3: MultiLayer Perceptron
* Try different activation function sigmoid, relu and tanh.
* In this part, I implement my own multilayer perceptron model for both regression and classification usage. It can be used with the [```MLPGenerator``` class and ```trainNN``` API](https://github.com/unlimitediw/CitiesPrediction/blob/master/MLPGenerator.py)
* The validation score of it is about 0.8, train score is about 0.3 with (9,6,1) on 10-cross validation and 100 iter. The training is slow with cross validation and different size so I may supplement the learning curve in the future.
* With 6 layer network (9,15,25,12,6,1) 100 iter. The validation score is 0.3853, The train score is 0.3365
* I found that set too many layer may let the model 
* With 8 layer network (9,15,25,35,24,18,12,6,1), the model can not converge and optimization ends immeadiately. It is due to too small dataset for a neural network.
* I also found that with more layer and higher scale for each layer, the result may fixed to some value. I havn't completed understand it now but I try to figure out last layers in my MLP. [My layer neuron checking](https://github.com/unlimitediw/CitiesPrediction/blob/master/CheckNN.py)


## Part 4: Convolutional Neural Network Regression
* Memory Error for 1280x1280 image processing (1280x1280x3 ndarray x 3953batch)
* Solution:
  - [AWS SageMaker](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances), [My AWS](https://github.com/unlimitediw/dist-sys-practice/blob/master/Technical_Report.md)
  - I will not use PCA because it is not aligned and CNN do this job with nonlinear activation layer[PCA1280x1280](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/PCA1280_1280.png), [PCA300x1280](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/PCA300_1280.png)
  #
      def getUSV(X):
        # covariance matrix formular
        cov_matrix = X.T.dot(X) / X.shape[0]
        U, S, V = scipy.linalg.svd(cov_matrix, full_matrices=True, compute_uv=True)
        return U, S, V


      # nice PCA 用新的单位向量去生成新的少量投影数据
      def projectData(X, U, K):
          # project only top "K" eigenvectors
          Ureduced = U[:, :K]
          z = X.dot(Ureduced)
          return z


      def recoverData(Z, U, K):
          Ureduced = U[:, :K]
          Xapprox = Z.dot(Ureduced.T)
          return Xapprox

  - 

0.91 to 0.935 for test and 0.93 for training data, very good.
[CNNCoding part](https://github.com/unlimitediw/CitiesPrediction/blob/master/mapToGDP.py)
[Architecture](https://www.researchgate.net/figure/llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means_fig2_325137356)
![Test Matrix](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/TestMatrix.png)
![Train Matrix](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/TrainMatrix.png)   

## Part 5: Combining Map Image Feature with Numeric Feature to Predict GDP.
Not start yet.

## Part 5: Map Generation with Cycle GAN
[Junyanz's Model](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Future Part:
* Due to lack of history continuous data, It is very hard to apply Recursive Neural Network to predict the future development. But now I also found it from OCED, so it may start in the future.
* I don't have enough time to do the full world map searching part to find the livable place.
* Apply It to AWS SageMaker. In my distributed course, I just learned how to use some techniques such as docker containers, spark and Sagemaker and I have tried to train and deployment some machine learning model on the website with SageMaker. In the future, I will combine it with this project.

## Reference:
1. http://cs229.stanford.edu/materials/smo.pdf
2. [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf)
3. [GAN](https://arxiv.org/pdf/1406.2661.pdf)

## Data Source
1. Google
2. [OECD](https://stats.oecd.org/Index.aspx?QueryId=51329#) 
3. [ergebnis](https://fingolas.carto.com/tables/ergebnis/public)

## What I learn from this project
* Google Cloud Platform API
* AWS SageMaker
* Multithreading in Python
* OOP programming in Python
* VGG 19 CNN
* Multilayer perceptron for regression
* Support Vector Regression
* PCA for RGB image
* Pandas and Numpy skills for data preprocessing

End
