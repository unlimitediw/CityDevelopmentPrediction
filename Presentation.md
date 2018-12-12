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
* There are also many other important features may help me to get higher accuracy in city GDP prediction. However, my focus points should be the model construction and optimization and is not familiar to deal with the deep city statistic such as "population by age" or "labour markert".
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
* With the name of 4054 cities (collected in the numeric part), I insert it into my google map API. The format is https://maps.googleapis.com/maps/api/staticmap?&center=beijing&zoom=10&format=png&maptype=roadmap&style=feature:road|visibility:off&style=element:labels%7Cvisibility:off&size=640x640&scale=2&key=MyKe
* The city name can also be replaced by coordinate likes (54.321,-12.345).
* With this url, you can easily adjust the format of map image you want. For instance, you can choose maptype = "roadmap" or "satellite" and adjust the zoom to get the scale you want for the city map.
* In my project, I coolect four kind of map data: roadmap, roadmap without road, roadmap withou map and the satellite map.
## Part 2: Support Vector Regression
* I implement my own [SMO function](https://github.com/unlimitediw/MLGWU/blob/master/ML/CS6364_HW3_SVM_Handwork.py) for weight tunning in this part. But still use the sklearn.svm to train my model for higher speed.
* After multiple C trying with 10-cross validation, I select C = 90000.
* rbf is basically better than poly
* Larger training set can avoid overfitting
* The score of it is about 0.43 for 10-cross validation. And the score function is below:
#
      score = abs(k[i] - yTest[i])/yTest[i])
      # I finally decided to use yTest rather than max(yTest,hypo) because it can better fit my lost function. The hypo can be small but shouldn't be too large. If use max(yTest,hypo) the validation error for both svm and MLP will be about 0.3 and will not change.
* Learning Curve with different C value:
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/SVMLcurveCValue.png)
* Learning Curve with different training data size (this curve is with seed = 3, I try many seeds, this is a bad one. However, at least it is a random choose point. The reason that 80% data validation error is larger than 60% one is that my dataset is too small (229 cities). There is a large flictuation for different random setting):
![](https://github.com/unlimitediw/CitiesPrediction/blob/master/ReportImages/SVMLcurveSize.png)



## Part 3: MultiLayer Perceptron
* In this part, I implement my own multilayer perceptron model for both regression and classification usage. It can be used with the [```MLPGenerator``` class and ```trainNN``` API](https://github.com/unlimitediw/CitiesPrediction/blob/master/MLPGenerator.py)
* The validation score of it is about 0.8, train score is about 0.3 with (9,6,1) on 10-cross validation and 100 iter. The training is slow with cross validation and different size so I may supplement the learning curve in the future.
* With 6 layer network (9,15,25,12,6,1) 100 iter. The validation score is 0.3853, The train score is 0.3365
* I found that set too many layer may let the model 
* With 8 layer network (9,15,25,35,24,18,12,6,1), the model can not converge and optimization ends immeadiately. It is due to too small dataset for a neural network.


## Part 4: Convolutional Neural Network Regression

## Part 5: Combining Map Image Feature with Numeric Feature to Predict GDP.

## Part 5: Map Generation with Cycle GAN

## Future Part:
* Due to lack of history continuous data, It is very hard to apply Recursive Neural Network to predict the future development.
* I don't have enough time to do the full world map searching part to find the livable place.
* Apply It to AWS SageMaker. In my distributed course, I just learned how to use some techniques such as docker containers, spark and Sagemaker and I have tried to train and deployment some machine learning model on the website with SageMaker. In the future, I will combine it with this project.

## Reference:
1. http://cs229.stanford.edu/materials/smo.pdf
2. [Cycle GAN](https://arxiv.org/pdf/1703.10593.pdf)
3. [GAN](https://arxiv.org/pdf/1406.2661.pdf)
4. 

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
* 
