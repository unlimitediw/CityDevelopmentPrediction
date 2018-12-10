# City Prediction
### Wentao Li
### unlimitediw@gwmail.gwu.edu

**************
### Abstract
There are [4037 cities](https://brilliantmaps.com/4037-100000-person-cities/) with more than 100,000 population around the world from the largest cities such as [Shanghai](https://en.wikipedia.org/wiki/Shanghai), [Tokyo](https://en.wikipedia.org/wiki/Tokyo) and [New York City](https://en.wikipedia.org/wiki/New_York_City) to some small cities like [Boise](https://en.wikipedia.org/wiki/Boise,_Idaho) and [Thunder Bay](https://en.wikipedia.org/wiki/Thunder_Bay) and also from some well developed historic cities like [London](https://en.wikipedia.org/wiki/London) and [Istanbul](https://en.wikipedia.org/wiki/Istanbul) to some 'young' developing cities such as [Abidjan](https://en.wikipedia.org/wiki/Abidjan) and [Dar es Salaam](https://en.wikipedia.org/wiki/Dar_es_Salaam). Small city can grow to a large one and the young city also can be a well developed city in the future. It is believable that there are some correlation among these cities and in this project I want to find the relationship between the city development status and  ground truth features such as population, climate and Map image. Furthermore, I will predict the development trend of the city and generate the future hypothetical map image base on the previous data analysis and GAN technique. At the same time, I may try to find the places around the world map that have the possibility to become a livable city base on the prebuilt model and some geography and climate features.

### Introduction
When considering to evaluate a city, people always compare it with their familiar hometown and the world famous cities. However, it is not accurate sometime due to the lack of commonness among these cities and evaluating it without considering some implicit features such as city elevation map and metropolis area effect. What if we considering this evaluation problem with an overall database and good combination of machine learning methods?

In this project, I will try both kernel SVM and MLP method to process the numerical features such as population, longitude and latitude and category features such as climate like 'tropical wet' and 'Marine west coast' and generate the development level of the city. At the same time, I will use CNN model to find the commonness of the city map, elevation map, satellite map and so on. All of the SVM, MLP and CNN model will be a supervised or semisupervised learning model with label such as [GDP](https://en.wikipedia.org/wiki/Gross_domestic_product) level.

Furthermore, I will combine the city data in history timeline(by RNN or Markov model) and city category (by feature clustering or GDP level) to generate the future features and development level of this city. With this future prediction result, I will also try to generate the city map in the future by GAN technique. On the other hand, if have time, I will also try to extract some features such as geography and climate information from prebuilt model and do a world map searching with an evaluate function which aims to find the most livable place for human to build a new city.

More specifically, in the part of classification problem with numeric and category features input, soft SVM with kernel is able to linearly separate these cities but it will be slow for training if use a large scale of dataset and the performance is not guaranteed base on the kernel function. To the MLP model, the process of model training is more implicit and may perform better if the dataset is large and sparse and it can also solve the non-linear separable problem. I will compare their performance in the experiment part later. In the part of map image processing, I select CNN method since it has a high performance in image data processing with weight sharing and feature extraction. For instance, CNN can recognize the river or coast line no matter which direction and shape it is. In the part of city future trend prediction, RNN or Markov model is more applicable due to their good performance in future prediction base on previous states.

The following two section about future map generating and livable place searching is more productive and interesting. In the part of map generating, I will use a unsupervised learning algorithm [GAN](https://arxiv.org/pdf/1406.2661.pdf)(Generative adversarial network) and more specifically use the [Image to Image Translation with a conditional generative adversarial network](https://arxiv.org/pdf/1611.07004.pdf) and [cycle-consistent adversarial network](https://arxiv.org/pdf/1703.10593.pdf). GAN algorithm includes two parts which one for generating fake image and one for discriminating the fake image from the real image and play a minmax game. In my model, I will first train the GAN model with data A,B that A has the real feature and real image while B has the real features and generates the fake image. If the model is trained successfully, I can use this model and the predicted features from RNN to generate the future map. May be we can have a view of the future world and it will work well especially in the developing countries since it can takes the developed countries data as a template. 

The second section is about the livable spaces searching and it is more like an application of my evaluation model. I will first do a region evaluation to find the high value and low human activity areas like plain and spilt these area into thousands of little square images with features of weather, elevation map and so on. After that, I will apply these features to my evaluation model and generate a list of hypo values. The places with highest value may be taken into consideration to build a city.

*****************************
## Part 1: City Data Preprocessing
* The 4054 cities population and coordinate data is from [ergebnis](https://fingolas.carto.com/tables/ergebnis/public) publiced in 2014
* The 313 cities GDP and corresponding feature (Georgraphic and administrative forms) data is from [OECD](https://stats.oecd.org/Index.aspx?QueryId=51329#) 
* There are also many other important features may help me to get higher accuracy in city GDP prediction. However, my focus points should be the model construction and optimization and is not familiar to deal with the deep city statistic such as "population by age" or "labour markert".
* My Data Preprocessing for combining these two .cvs file is [datapreprocessing](https://github.com/unlimitediw/DataSearch-Preprocessing/blob/master/DataPreprocessing.py)
* The data preprocessing includes:
  1. Removing Nan value.
  2. Mergeing city feature with city gdp, population and coordinate with city name regex and finally get 229 cities data.
  3. Changing all feature data type into numpy.float64.
  4. Save the new cities data in .csv format for future usage.
  (The feature normalization part is included in the model construction part)
  
## Part 2: Support Vector Regression
* Use standard normalization to all features except coordinate feature and number counting feature.
#
      def stdScl(F):
        F = (F - np.average(F))/np.std(F)
        return F


