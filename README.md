# On-Time-Ticket-Payment-Predictive-Model
ML Model to predict whether a given blight ticket will be paid on time.

Python file contains the function that trains the On-Time Ticket Payment Predictive Model to predict blight ticket compliance in Detroit using readonly/train.csv. 
This model returns a series of length 61001 with the data being the probability that each corresponding ticket from readonly/test.csv will be paid, and the index being the ticket_id.


File descriptions:

readonly/train.csv - the training set (all tickets issued 2004-2011)
readonly/test.csv - the test set (all tickets issued 2012-2016)
readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
 Note: misspelled addresses may be incorrectly geolocated.


The two data files for use in training and validating my model: train.csv and test.csv. 
Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. 
The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
