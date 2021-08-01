# AIIntoMushrooms

The main file for this project is mushroom_labeling. *Enter*
running it will perform all desired methods and plot all relevant graphs *Enter*
the desired functionalities can be configured in the start of the script under the comment: "desired functionalities" (for example - running supervised learining/clustering)*Enter**
possoble funcionalities:**Enter**
  should_reduce_features - if True, uses PCA to reduce the number of dimentions to the size specified in "number_of_features"
  should_use_clustering - if True, running the clustering models and plots the relevant results
  should_use_supervised_learning - if True, running the supervised learning models and plots the relevant results
  should_present_variance_explained - if True, will show the variance explained graph for the data
  clusters_nums - the number of clusters to run the clustering models on.
  number_of_features - the number of features to reduce to dimension to if "should_reduce_features" is set to True
  

all other files are files that mushrooms_labeling.py calls them:
data_prepare - preparing the data for running the models on it
neural_network - the neural network created for the problem
supervised_learning - using the neural network and the other supervised learning algorithms to solve the problem
clustering - using the clustering algorithms to solve to problem
plot_silhouette - plots the silhouettes graphs for the clustering methods
compare_methods - presenting confusion matrices and calculates recall scores
