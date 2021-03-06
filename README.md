# Credit Risk.Machine learning application.



## Abstract 

This is my case studies for my thesis in Banking and Finance about credit scoring.

This work aim to, on one side, the backend code develop on Python Notebook (code_streamlit_np.ipynb), for reason of readability, to find the best Hyperparameter in order to achieve the best forecast for the credit scoring with algorithm that i analize during the thesis, in particular:

**1)Logistic regression and Penalized logistic regression (ridge and lasso)**

**2)Lda**

**3)Random Forest**

**4)Neural Network**

on the other side the code for the web application Streamlit, that is deployed on internet with Heroku, to allow to perform the same analysis without previous knowladge of pythone and coding:

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
"black box" about variable and also perform the same analysis without coding.

the database is stored at this link : https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./releases/download/credit_scoring.ml/LendingClub.csv 
since there is problem to find a good database we use LendingClub that use variable that we don't suggest to use, in order to avoid "no ethical scoring".

By the way the project only aim to show how the algorithm performs on the given data for find the best one

In the end the final result optimized with cross validation of Hyperparameters show this:

![alt text](https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/image09.png)

That Random Forest is the best algorithm for Unbalance dataset

stored at this link : https://github.com/fedeghigo/Credit-Risk.Machine-learning-application./blob/master/code_streamlit_np.ipynb
