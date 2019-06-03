Fisher’s Iris Classification Problem
This is a classical benchmark problem in pattern recognition which asks to classify the 3 species of plants (iris) from the characteristics observed in their flowers.
Reference : Fisher, R.A. (1936). "The Use of Multiple Measurements in Taxonomic Problems". Annals of Eugenics 7: 179–188.
Classes (3 classes)
1. Iris Setosa 2. Iris Versicolour 3. Iris Virginica
Features
1. Petal length (cm), 2. Petal width (cm), 3. Sepal length (cm), 4. Sepal width (cm)
Data
50 examples/class. 150 examples in total.
Problem
Download the Fisher’s iris data from the Machine Learning Repository at UC Irvine
http://archive.ics.uci.edu/ml/datasets/Iris
In the file “iris.data”, each row represents a measurement from an iris flower. Each row include the 4 features and the species (class).
Divide the data of each class into half. Use one half for training (training set), and the other half for testing (test set). Build classifiers for this data by the following two methods A and B.
A. Optimal Linear Associative Memory (OLAM).
B. Fisher’s linear discriminator. Prepare 3 classifiers for 2-class problems classifying one class against the other two. You may also be interested to seek information about the Multi-class Linear Discriminant Analysis in the literature.
The classifiers should be trained using the training set, and evaluated using the test set. Repeat (1) random selection of training set, (2) training, and (3) testing for 20 times. Evaluate the mean correct classification rates and their variances for both methods.
You may use the programming language of your choice. Try programing the core classifier parts by yourself (do not use the ready-made classifier functions), and see if it really works as explained in the lecture. In the report, include your source codes.
