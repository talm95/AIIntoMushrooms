# AIIntoMushrooms

The main file for this project is mushroom_labeling.<br />
running it will perform all desired methods and plot all relevant graphs<br />
the desired functionalities can be configured in the start of the script under the comment: "desired functionalities" (for example - running supervised learining/clustering). <br />
possoble funcionalities:  <br />
  should_reduce_features - if True, uses PCA to reduce the number of dimentions to the size specified in "number_of_features". <br />
  should_use_clustering - if True, running the clustering models and plots the relevant results. <br />
  should_use_supervised_learning - if True, running the supervised learning models and plots the relevant results. <br />
  should_present_variance_explained - if True, will show the variance explained graph for the data  <br />
  clusters_nums - the number of clusters to run the clustering models on.  <br />
  number_of_features - the number of features to reduce to dimension to if "should_reduce_features" is set to True. <br />
  

all other files are files that mushrooms_labeling.py calls them:  <br />
data_prepare - preparing the data for running the models on it. <br />
neural_network - the neural network created for the problem. <br />
supervised_learning - using the neural network and the other supervised learning algorithms to solve the problem. <br />
clustering - using the clustering algorithms to solve to problem. <br />
plot_silhouette - plots the silhouettes graphs for the clustering methods. <br />
compare_methods - presenting confusion matrices and calculates recall scores. <br />
