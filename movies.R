# Loading the sufficient dataset and removing the NA values

library(randomForest)
library(gbm)
library(e1071)
library(MASS)
library(ISLR)
require(ISLR)
library(class)
require(class)
require(boot)
require(tree)
movies = read.csv("./movies.csv", na.strings = "?")
movies = na.omit(movies)
attach(movies)
totalrows = dim(movies)[1]
imdb_rating = rep("bad", totalrows)

# Classifying the predictor based on imdb_score: 
# If less than 6 then classifying it as bad movie
# If between 6 and 8 then classifying it as average movie
# If between 8 and 10 then classifying it as good movie

imdb_rating[imdb_score<6] = "bad"
imdb_rating[(imdb_score>=6) & (imdb_score<8)] = "average"
imdb_rating[(imdb_score>=8)] = "good"

# Making a new data set for movie

moviesdataset = data.frame(movies, imdb_rating)
#fix(moviesdataset)
hist(imdb_score, col=5, breaks =15)
summary(moviesdataset)
summary(moviesdataset$imdb_rating)
# Sampling the data set into 70:30 ratio for Training and Testing purpose

percent70data = round(0.70*totalrows)
percent30data = totalrows - percent70data
train = 1:percent70data
test = (percent70data+1):totalrows
moviesTrain = moviesdataset[train, ]
moviesTest = moviesdataset[test, ]
dim(moviesTrain)
dim(moviesTest)

# Plots with all predictors

attach(moviesdataset)
skewness(imdb_score)
pairs(~imdb_score+num_critic_for_reviews+director_facebook_likes+gross+num_voted_users)
pairs(~imdb_score+facenumber_in_poster+num_user_for_reviews+movie_facebook_likes)
plot(imdb_score, num_critic_for_reviews)
plot(imdb_score, director_facebook_likes)
plot(imdb_score, gross)
plot(imdb_score, num_voted_users)
plot(imdb_score, facenumber_in_poster)
plot(imdb_score, num_user_for_reviews)
plot(country, imdb_score)
plot(imdb_score, movie_facebook_likes)
plot(imdb_score, budget)                    # Impacting not at all
plot(imdb_score, cast_total_facebook_likes) # Impacting very few
plot(imdb_score, duration)                  # Impacting not at all
plot(imdb_score, actor_2_facebook_likes)    # Impacting very few
plot(imdb_score, actor_1_facebook_likes)    # Impacting not at all
plot(imdb_score, actor_3_facebook_likes)    # Impacting not at all
plot(color, imdb_score)                     # Impacting not at all
plot(director_name, imdb_score)             # Impacting not at all

# We are trying with Multiple Linear Regression Model based on the above plots for determination of predictors to come up with the predicted imdb score
par(mfrow=c(2,2))
dim(moviesdataset)
glm.fit1 = lm(imdb_score~num_critic_for_reviews + actor_1_facebook_likes + actor_2_facebook_likes + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + country + movie_facebook_likes, data=moviesTrain)
summary(glm.fit1)
plot(glm.fit1)
glm.fit2 = lm(imdb_score~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + country + movie_facebook_likes, data=moviesTrain)
summary(glm.fit2)
plot(glm.fit2)
glm.fit3 = lm(imdb_score~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + movie_facebook_likes, data=moviesTrain)
summary(glm.fit3)
plot(glm.fit3)
glm.fit4 = lm(imdb_score~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, data=moviesTrain)
summary(glm.fit4)
plot(glm.fit4)
glm.fit5 = lm(imdb_score~num_critic_for_reviews + movie_facebook_likes*director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, data=moviesTrain)
summary(glm.fit5)
plot(glm.fit5)

# Having cross validation on the above predicted Linear Regression models having high R square

par(mfrow=c(1,1))
glm.fit3CV = glm(imdb_score~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + movie_facebook_likes, data=moviesTrain)
glm.fit4CV = glm(imdb_score~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, data=moviesTrain)
glm.fit5CV = glm(imdb_score~num_critic_for_reviews + movie_facebook_likes*director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, data=moviesTrain)
cv.glm(moviesTrain, glm.fit3CV)$delta[1]
cv.glm(moviesTrain, glm.fit4CV)$delta[1]
cv.glm(moviesTrain, glm.fit5CV)$delta[1]
cv.glm(moviesTrain, glm.fit3CV, K=5)$delta[1]
cv.glm(moviesTrain, glm.fit4CV, K=5)$delta[1]
cv.glm(moviesTrain, glm.fit5CV, K=5)$delta[1]
cv.glm(moviesTrain, glm.fit3CV, K=10)$delta[1]
cv.glm(moviesTrain, glm.fit4CV, K=10)$delta[1]
cv.glm(moviesTrain, glm.fit5CV, K=10)$delta[1]

# Running the best fitted models (Multiple Linear Regression statistical learning method) on test data set; We found glm.fit4 and glm.fit5 models are better ones which can be predicted on Test Data set

glm.predict3conf = confint(glm.fit3)
glm.predict3conf
glm.predict4conf = confint(glm.fit4)
glm.predict4conf
glm.predict5conf = confint(glm.fit5)
glm.predict5conf
glm.predict3 = predict(glm.fit3, moviesTest)
summary(glm.predict3)
glm.predict4 = predict(glm.fit4, moviesTest)
summary(glm.predict4)
glm.predict5 = predict(glm.fit5, moviesTest)
summary(glm.predict5)
glm.prob3 = rep("bad", nrow(moviesTest))
glm.prob3[(predict(glm.fit3) < 6)] = "bad"
glm.prob3[(predict(glm.fit3) >= 6) & (predict(glm.fit3) < 8)] = "average"
glm.prob3[(predict(glm.fit3) >= 8)] = "good"
mean(glm.prob3 != moviesTest$imdb_rating)
glm.prob4 = rep("bad", nrow(moviesTest))
glm.prob4[(predict(glm.fit4) < 6)] = "bad"
glm.prob4[(predict(glm.fit4) >= 6) & (predict(glm.fit4) < 8)] = "average"
glm.prob4[(predict(glm.fit4) >= 8)] = "good"
mean(glm.prob4 != moviesTest$imdb_rating)
glm.prob5 = rep("bad", nrow(moviesTest))
glm.prob5[(predict(glm.fit5) < 6)] = "bad"
glm.prob5[(predict(glm.fit5) >= 6) & (predict(glm.fit5) < 8)] = "average"
glm.prob5[(predict(glm.fit5) >= 8)] = "good"
mean(glm.prob5 != moviesTest$imdb_rating)

# As of now, model 4 (glm.fit4) from Multiple Linear Regression makes more fit on testing data with 38.74% test error rate while model 5 (glm.fit5) makes more than 40.77% test error rate
# Adjusted R square value for model 4 was little low than model 5; RSS was high for model 4 compared to model 5

# We are not using the Logistic Regression models as we are classifying our target based on 4 levels [bad, average, good and best]
# Using the LDA model now with only those predictors which are highly associated with the target variable; We are using the predictors of glm.fit4 and glm.fit5 models from the earlier stage

attach(moviesTrain)
lda.fit3 = lda(imdb_rating~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + movie_facebook_likes, data=moviesTrain)
lda.fit3
plot(lda.fit3)
lda.fit4 = lda(imdb_rating~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, data=moviesTrain)
lda.fit4
plot(lda.fit4)
lda.fit5 = lda(imdb_rating~num_critic_for_reviews + movie_facebook_likes*director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, data=moviesTrain)
lda.fit5
plot(lda.fit5)
# Use the above fitted LDA model on test data set

lda.predict3 = predict(lda.fit3, moviesTest)
table(lda.predict3$class, moviesTest$imdb_rating)
mean(lda.predict3$class != moviesTest$imdb_rating)
lda.predict4 = predict(lda.fit4, moviesTest)
table(lda.predict4$class, moviesTest$imdb_rating)
mean(lda.predict4$class != moviesTest$imdb_rating)
lda.predict5 = predict(lda.fit5, moviesTest)
table(lda.predict5$class, moviesTest$imdb_rating)
mean(lda.predict5$class != moviesTest$imdb_rating)

# We can clearly see from LDA models that model 4 which we have used in Multiple Linear Regression model performed well on testing data set as it provided 27.9% test error rate while model 5 provided 28% test error rate

# Using the KNN model now with only those predictors which are highly associated with the target variable; We are using the predictors of glm.fit4 and glm.fit5 models from the very earlier stage

set.seed(1)
attach(moviesdataset)
knntrain = as.matrix(moviesdataset[train, ])
knntest = as.matrix(moviesdataset[test, ])
predictors = cbind(num_critic_for_reviews,director_facebook_likes,gross,num_voted_users,facenumber_in_poster,num_user_for_reviews)
p = imdb_rating[train]
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=5)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=10)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=15)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=20)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
predictors = cbind(num_critic_for_reviews,movie_facebook_likes,director_facebook_likes,gross,num_voted_users,facenumber_in_poster,num_user_for_reviews)
p = imdb_rating[train]
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=5)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=10)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=15)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=20)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])

# K=20 performs better result on both the models which provides lower test error rate, 34.21%

# Using Single Decision Tree model now on the above data set

set.seed(1)
#tree.movies=tree(imdb_rating~num_critic_for_reviews + director_facebook_likes + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews, moviesdataset[train,])
tree.movies=tree(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score, moviesdataset[train,])
plot(tree.movies);text(tree.movies,pretty=0)
tree.pred=predict(tree.movies,moviesdataset[-train,],type="class")
with(moviesdataset[-train,],table(tree.pred,imdb_rating))
summary(tree.movies)
(293+3+111+3+49)/1140
# Very high error rate from the tree model
# Trying to use cross validation to get the best sequence of subtrees and then pruning the tree to fit on the best subtree 
cv.movies=cv.tree(tree.movies,FUN=prune.misclass)
cv.movies
plot(cv.movies)
prune.movies=prune.misclass(tree.movies,best=5) 
#Even tried with 5 and 6 as suggested by cv.movies but results are same
plot(prune.movies);text(prune.movies,pretty=0)
summary(prune.movies)
tree.pred=predict(prune.movies,moviesdataset[-train,],type="class")
with(moviesdataset[-train,],table(tree.pred,imdb_rating))
(293+3+111+3+49)/1140

# Plots for Project Progress Report for Accuracy of different models 
x=c("MultileLM","LDA","KNN (k=20)","SDT", "Random Forest")
x=as.factor(x)
y=c(0.62, 0.72, 0.67, 0.60, 0.72)
plot(x,y,xlab="Models Introduced", ylab="Accuracy",main="Accuracy versus Models")
# legend(4.5,0.71, c("SDT is Single Decision Tree"))

x=c(5,10,15,20)
y=c(0.62, 0.63, 0.66, 0.67)
plot(x,y,xlab="k values", ylab="Accuracy",main="Comparing Accuracy in KNN Model for k values", col="red", type="b")



x=c("KNN (old)","KNN (model a)","KNN (model b)")
x=as.factor(x)
y=c(0.66, 0.65, 0.645)
plot(x,y,xlab="All KNN Models", ylab="Accuracy",main="Comparing Accuracy in all KNN Models")


x=c("Old","New")
x=as.factor(x)
y=c(0.49, 0.60)
plot(x,y,xlab="All Single Tree Models", ylab="Accuracy",main="Comparing Accuracy in all Single Models")

# Trying Random Forest Approach on 14 predictors

set.seed(1)
#fix(moviesdataset)
randomForest.movies=randomForest(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=moviesdataset,subset=train,mtry=6,importance=TRUE)
yhat.rf=predict(randomForest.movies, newdata=moviesdataset[-train ,])
movies.test=moviesdataset[-train ,"imdb_rating"]
table(yhat.rf,movies.test)
mean(yhat.rf==movies.test)
importance(randomForest.movies)
randomForest.movies=randomForest(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=moviesdataset,subset=train,mtry=7,importance=TRUE)
yhat.rf=predict(randomForest.movies, newdata=moviesdataset[-train ,])
movies.test=moviesdataset[-train ,"imdb_rating"]
table(yhat.rf,movies.test)
mean(yhat.rf==movies.test)
importance(randomForest.movies)
randomForest.movies=randomForest(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=moviesdataset,subset=train,mtry=5,importance=TRUE)
yhat.rf=predict(randomForest.movies, newdata=moviesdataset[-train ,])
movies.test=moviesdataset[-train ,"imdb_rating"]
mean(yhat.rf==movies.test)
importance(randomForest.movies)

x=c(5,6,7)
y=c(0.7113,0.7194,0.7201)
plot(x,y,xlab="mtry in Random Forest", ylab="Accuracy", main="Accuracy in Random Forest models for different mtry", col="blue", type="b")

# Trying the Boosting Approach

set.seed(1)
boost.movies=gbm(imdb_score~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=moviesdataset[train ,], distribution="gaussian", n.trees=5000, interaction.depth=4, verbose=F)
summary(boost.movies)
names(boost.movies)
plot(boost.movies ,i="num_voted_users")
plot(boost.movies ,i="duration")
plot(boost.movies ,i="budget")
plot(boost.movies ,i="num_user_for_reviews")
yhat.boost=predict(boost.movies ,newdata=moviesdataset[-train ,], n.trees=5000)
boost.moviesTest = moviesdataset[-train,"imdb_score"]
mean((yhat.boost - boost.moviesTest)^2)

# High MSE obtained of around 0.786 in Boosting; See if we can make it lower

set.seed(1)
attach(moviesdataset)
knntrain = as.matrix(moviesdataset[train, ])
knntest = as.matrix(moviesdataset[test, ])
predictors = cbind(num_critic_for_reviews,director_facebook_likes,num_critic_for_reviews * director_facebook_likes,gross,num_voted_users,facenumber_in_poster,num_user_for_reviews, num_voted_users * num_user_for_reviews)
p = imdb_rating[train]
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=5)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=10)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=15)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=20)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
predictors = cbind(num_critic_for_reviews,movie_facebook_likes,director_facebook_likes,gross,num_voted_users,facenumber_in_poster,num_user_for_reviews)
p = imdb_rating[train]
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=5)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=10)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=15)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
knn.pred1 = knn(predictors[train,],predictors[test,],p,k=20)
table(knn.pred1,imdb_rating[test])
mean(knn.pred1!=imdb_rating[test])
# with combinations added, prediction1 is the best with error rate being 33.33%


# Oversampled data set Prediction

movies = read.csv("./movies.csv", na.strings = "?")
movies = na.omit(movies)
attach(movies)
totalrows = dim(movies)[1]

# Classifying the predictor based on imdb_score: 
# If less than 6 then classifying it as bad movie
# If between 6 and 8 then classifying it as average movie
# If between 8 and 10 then classifying it as good movie

imdb_rating = rep("bad", totalrows)
imdb_rating[imdb_score<6] = "bad"
imdb_rating[(imdb_score>=6) & (imdb_score<8)] = "average"
imdb_rating[(imdb_score>=8)] = "good"
moviesdataset = data.frame(movies, imdb_rating)

# Sampling the data set into 70:30 ratio for Training and Testing purpose
averages = subset(moviesdataset, imdb_rating == 'average')
dim(averages)
bads = subset(moviesdataset, imdb_rating == 'bad')
dim(bads)
goods = subset(moviesdataset, imdb_rating == 'good')
dim(goods)
oversampled_goods = goods[sample(nrow(goods), 2000, replace = TRUE),  ]
oversampled_bads = bads[sample(nrow(bads), 2000, replace = TRUE), ]
dim(oversampled_bads)
dim(oversampled_goods)
goods_and_bads_dataset = merge(oversampled_goods, oversampled_bads, all.x = TRUE, all.y = TRUE)
dim(goods_and_bads_dataset)
final_dataset = merge (goods_and_bads_dataset, averages, all.x  = TRUE, all.y = TRUE)
dim(final_dataset)
totalrows = dim(final_dataset)[1]
attach(final_dataset)
percent70data = round(0.70*totalrows)
percent30data = totalrows - percent70data
train = 1:percent70data
test = (percent70data+1):totalrows
moviesTrain = final_dataset[train, ]
moviesTest = final_dataset[test, ]
dim(moviesTrain)
dim(moviesTest)
hist(imdb_score, col=5, breaks =15)
summary(final_dataset)
summary(final_dataset$imdb_rating)

# We are trying with Multiple Linear Regression Model
set.seed(1)
glm.fit6 = lm(imdb_score~num_critic_for_reviews + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + movie_facebook_likes, data=moviesTrain)
summary(glm.fit6)
glm.predict6 = predict(glm.fit6, moviesTest)
summary(glm.predict6)
glm.prob6 = rep("bad", nrow(moviesTest))
glm.prob6[(glm.predict6 < 6)] = "bad"
glm.prob6[(glm.predict6 >= 6) & (glm.predict6 < 8)] = "average"
glm.prob6[(glm.predict6 >= 8)] = "good"
mean(glm.prob6 != moviesTest$imdb_rating)

# Using the LDA model now with only those predictors which are highly associated with the target variable;
lda.fit6 = lda(imdb_rating~num_critic_for_reviews + gross + num_voted_users + facenumber_in_poster + num_user_for_reviews + movie_facebook_likes, data=moviesTrain)
lda.fit6
plot(lda.fit6)
lda.predict6 = predict(lda.fit6, moviesTest)
table(lda.predict6$class, moviesTest$imdb_rating)
mean(lda.predict6$class != moviesTest$imdb_rating)

# Using Single Decision Trees
set.seed(1)
tree.movies=tree(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score, final_dataset[train,])
plot(tree.movies);text(tree.movies,pretty=0)
tree.pred=predict(tree.movies,final_dataset[-train,],type="class")
with(final_dataset[-train,],table(tree.pred,imdb_rating))
(165+114+224+7+107)/1959
cv.movies=cv.tree(tree.movies,FUN=prune.misclass)
cv.movies
plot(cv.movies)
prune.movies=prune.misclass(tree.movies,best=12) 
plot(prune.movies);text(prune.movies,pretty=0)
summary(prune.movies)
tree.pred=predict(prune.movies,final_dataset[-train,],type="class")
with(final_dataset[-train,],table(tree.pred,imdb_rating))

# Trying Random Forest Approach on 14 predictors
set.seed(1)
randomForest.movies=randomForest(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=final_dataset,subset=train,mtry=6,importance=TRUE)
yhat.rf=predict(randomForest.movies, newdata=final_dataset[-train ,])
movies.test=final_dataset[-train ,"imdb_rating"]
table(yhat.rf,movies.test)
mean(yhat.rf==movies.test)
importance(randomForest.movies)
randomForest.movies=randomForest(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=final_dataset,subset=train,mtry=7,importance=TRUE)
yhat.rf=predict(randomForest.movies, newdata=final_dataset[-train ,])
movies.test=final_dataset[-train ,"imdb_rating"]
mean(yhat.rf==movies.test)
table(yhat.rf,movies.test)
importance(randomForest.movies)
randomForest.movies=randomForest(imdb_rating~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=final_dataset,subset=train,mtry=5,importance=TRUE)
yhat.rf=predict(randomForest.movies, newdata=final_dataset[-train ,])
movies.test=final_dataset[-train ,"imdb_rating"]
mean(yhat.rf==movies.test)
importance(randomForest.movies)
x=c(5,6,7)
y=c(0.6513,0.6564,0.6569)
plot(x,y,xlab="mtry in Random Forest", ylab="Accuracy", main="Accuracy in Random Forest models for different mtry", col="blue", type="b")

# Trying the Boosting Approach
set.seed(1)
boost.movies=gbm(imdb_score~.-country-director_name-color-actor_2_name-genres-actor_1_name-movie_title-actor_3_name-plot_keywords-movie_imdb_link-language-content_rating-imdb_rating-imdb_score,data=final_dataset[train ,], distribution="gaussian", n.trees=5000, interaction.depth=4, verbose=F)
summary(boost.movies)
names(boost.movies)
plot(boost.movies ,i="num_voted_users")
plot(boost.movies ,i="duration")
plot(boost.movies ,i="budget")
plot(boost.movies ,i="num_user_for_reviews")
yhat.boost=predict(boost.movies ,newdata=final_dataset[-train ,], n.trees=5000)
boost.moviesTest = final_dataset[-train,"imdb_score"]
mean((yhat.boost - boost.moviesTest)^2)

attach(final_dataset)
skewness(imdb_score)