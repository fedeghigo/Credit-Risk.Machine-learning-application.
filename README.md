# Credit Risk.Machine learning application.



## Abstract 

This is my case studies for my thesis in Banking and Finance about credit scoring.

This work aim to, on one side, the backend code develop on Python Notebook (code_credit_scoring_np.ipynb), for reason of readability, to find the best Hyperparameter in order to achieve the best forecast for the credit scoring with algorithm we analize during the thesis, in particular:

-Logistic regression and Penalized logistic regression (ridge and lasso)
-Lda
-Random Forest

on the other side the code for the web application Streamlit, that is deployed on internet with Heroku

web link: https://creditscoring-ml-thesis.herokuapp.com/

the file:
-Procfile 
-Requirements

are required for allow heroku to deploy from GitHub

## How to use the web app:

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image.png)

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image02.png)

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image03.png)

**You will have some result as the following:**



![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image04.png)

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image05.png)

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image06.png)


### In the end:

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image07.png)
![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image08.png)


## Conclusion:
This project aim to foster transparency and how is possible to do a basic credit scoring and made it available for avoid 
"black box" to the public and also be aware of the variable used.
the database is stored at this link : https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./releases/download/credit_scoring.ml/LendingClub.csv 
since there is problem to find a good database we use LendingClub that use variable that we don't suggest to use to, in order to avoid "no ethical scoring".

By the way the project only aim to show how the algorithm performs on the given data for find the best one

In the end the final result optimized with cross validation of Hyperparameters show this:

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image09.png)
