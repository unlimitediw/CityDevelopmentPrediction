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


There are [4037 cities](https://brilliantmaps.com/4037-100000-person-cities/) with more than 100,000 population around the world from the largest cities such as [Shanghai](https://en.wikipedia.org/wiki/Shanghai), [Tokyo](https://en.wikipedia.org/wiki/Tokyo) and [New York City](https://en.wikipedia.org/wiki/New_York_City) to some small cities like [Boise](https://en.wikipedia.org/wiki/Boise,_Idaho) and [Thunder Bay](https://en.wikipedia.org/wiki/Thunder_Bay) and also from some well developed historic cities like [London](https://en.wikipedia.org/wiki/London) and [Istanbul](https://en.wikipedia.org/wiki/Istanbul) to some 'young' developing cities such as [Abidjan](https://en.wikipedia.org/wiki/Abidjan) and [Dar es Salaam](https://en.wikipedia.org/wiki/Dar_es_Salaam). Small city can grow to a large one and the young city also can be a well developed city in the future. It is believable that there are some correlation among these cities and in this project I want to find the relationship between the city development status and  ground truth features such as population, climate and Map image. Furthermore, I will predict the development trend of the city and generate the future hypothetical map image base on the previous data analysis and GAN technique. At the same time, I may try to find the places around the world map that have the possibility to become a livable city base on the prebuilt model and some geography and climate features.

<a name = "1"></a>
### Introduction [next section](#2)
When considering to evaluate a city, people always compare it with their familiar hometown and the world famous cities. However, it is not accurate sometime due to the lack of commonness among these cities and evaluating it without considering some implicit features such as city elevation map and metropolis area effect. What if we considering this evaluation problem with an overall database and good combination of machine learning methods?

In this project, I will try both kernel SVM and MLP method to process the numerical features such as population, longitude and latitude and category features such as climate like 'tropical wet' and 'Marine west coast' and generate the development level of the city. At the same time, I will use CNN model to find the commonness of the city map, elevation map, satellite map and so on. All of the SVM, MLP and CNN model will be a supervised or semisupervised learning model with label such as [GDP](https://en.wikipedia.org/wiki/Gross_domestic_product) level.

Furthermore, I will combine the city data in history timeline(by RNN or Markov model) and city category (by feature clustering or GDP level) to generate the future features and development level of this city. With this future prediction result, I will also try to generate the city map in the future by GAN technique. On the other hand, if have time, I will also try to extract some features such as geography and climate information from prebuilt model and do a world map searching with an evaluate function which aims to find the most livable place for human to build a new city.

More specifically, in the part of classification problem with numeric and category features input, soft SVM with kernel is able to linearly separate these cities but it will be slow for training if use a large scale of dataset and the performance is not guaranteed base on the kernel function. To the MLP model, the process of model training is more implicit and may perform better if the dataset is large and sparse and it can also solve the non-linear separable problem. I will compare their performance in the experiment part later. In the part of map image processing, I select CNN method since it has a high performance in image data processing with weight sharing and feature extraction. For instance, CNN can recognize the river or coast line no matter which direction and shape it is. In the part of city future trend prediction, RNN or Markov model is more applicable due to their good performance in future prediction base on previous states.

The following two section about future map generating and livable place searching is more productive and interesting. In the part of map generating, I will use a unsupervised learning algorithm [GAN](https://arxiv.org/pdf/1406.2661.pdf)(Generative adversarial network) and more specifically use the [Image to Image Translation with a conditional generative adversarial network](https://arxiv.org/pdf/1611.07004.pdf) and [cycle-consistent adversarial network](https://arxiv.org/pdf/1703.10593.pdf). GAN algorithm includes two parts which one for generating fake image and one for discriminating the fake image from the real image and play a minmax game. In my model, I will first train the GAN model with data A,B that A has the real feature and real image while B has the real features and generates the fake image. If the model is trained successfully, I can use this model and the predicted features from RNN to generate the future map. May be we can have a view of the future world and it will work well especially in the developing countries since it can takes the developed countries data as a template. 

The second section is about the livable spaces searching and it is more like an application of my evaluation model. I will first do a region evaluation to find the high value and low human activity areas like plain and spilt these area into thousands of little square images with features of weather, elevation map and so on. After that, I will apply these features to my evaluation model and generate a list of hypo values. The places with highest value may be taken into consideration to build a city.

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
* With the name of 4054 cities (collected in the numeric part), I insert it into my google map API. The format is https://maps.googleapis.com/maps/api/staticmap?&center=beijing&zoom=10&format=png&maptype=roadmap&style=feature:road|visibility:off&style=element:labels%7Cvisibility:off&size=640x640&scale=2&key=MyKey
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
