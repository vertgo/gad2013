# load required library
library(class)
library(ggplot2)
#################################################
# PREPROCESSING
#################################################

data <- iris                # create copy of iris dataframe
labels <- data$Species      # store labels
data$Species <- NULL        # remove labels from feature set (note: could
                            # alternatively use neg indices on column index in knn call)

#################################################
# TRAIN/TEST SPLIT
#################################################

set.seed(1)         # initialize random seed for consistency
                    # NOTE -- run for various seeds --> need for CV!

numFolds <- 10 #N Folds Five

N <- nrow(data)     # total number of records (150)
fold.size <- N/numFolds #well, we know this is gonna be an even split

shuffled.index <- sample(1:N, N)       # shuffle some indices, essentially
print( shuffled.index)

#################################################
# APPLY MODEL
#################################################

err.rates <- data.frame()       # initialize results object

max.k <- 100

fold.offset <- 1 # set the number offset to 0
for ( curFold in 1:numFolds)
{

	testSlice.index <- shuffled.index[ fold.offset : (fold.offset + fold.size-1) ] 
	cat('\n', 'testSlice.index = ', testSlice.index, ', fold number = ', curFold, '\n', sep='')     # print slice
	print( testSlice.index)
	train.data <- data[-testSlice.index, ]       # perform train/test split
	test.data <- data[testSlice.index, ]       # note use of neg index...different than Python!

	train.labels <- as.factor(as.matrix(labels)[-testSlice.index, ])     # extract training set labels
	test.labels <- as.factor(as.matrix(labels)[testSlice.index, ])     # extract test set labels
	
	for (k in 1:max.k)              # perform fit for various values of k
	{
	    knn.fit <- knn(train = train.data,          # training set
	                    test = test.data,           # test set
	                    cl = train.labels,          # true labels
	                    k = k                       # number of NN to poll
	               )

	    cat('\n', 'k = ', k, ', fold number = ', curFold, '\n', sep='')     # print params
	    print(table(test.labels, knn.fit))          # print confusion matrix

	    this.err <- sum(test.labels != knn.fit) / length(test.labels)    # store gzn err

	    err.rates <- rbind(err.rates, this.err)     # append err to total results
	}

	fold.offset <- fold.offset + fold.size #increment the current fold offset number so we sample the next slice
}

#################################################
# OUTPUT RESULTS
#################################################

results <- data.frame(1:max.k, err.rates)   # create results summary data frame
names(results) <- c('k', 'err.rate')        # label columns of results df

# create title for results plot
title <- paste('knn results (numFolds = ', numFolds, ')', sep='')

# create results plot
results.plot <- ggplot(results, aes(x=k, y=err.rate)) + geom_point() + geom_line()
results.plot <- results.plot + ggtitle(title)

# draw results plot (note need for print stmt inside script to draw ggplot)
print(results.plot)

#################################################
# NOTES
#################################################

# what happens for high values (eg 100) of max.k? have a look at this plot:
# > results.plot <- ggplot(results, aes(x=k, y=err.rate)) + geom_smooth()

# our implementation here is pretty naive, meant to illustrate concepts rather
# than to be maximally efficient...see alt impl in DMwR package (with feature
# scaling):
#
# > install.packages('DMwR')
# > library(DMwR)
# > knn

# ed. note: how not to do it (black box)
# http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Classification/kNN

# R docs
# http://cran.r-project.org/web/packages/class/class.pdf
# http://cran.r-project.org/web/packages/DMwR/DMwR.pdf
