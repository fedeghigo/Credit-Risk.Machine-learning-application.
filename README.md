<p align="center">
  <img src="https://raw.githubusercontent.com/fedeghigo/Credit-Risk.Machine-learning-application./master/4459b235d8034836bbf2db266f30baa0__.png" />
</p>
<p align="center">
  <h1 align="center">
   Credit Risk Machine learning application
</h1>
<h4 align="center">:star: my repo so you can keep updated!</h4> 
</p>


## Abstract 

This is my case studies for my thesis in Banking and Finance about credit scoring.

This work aim to, on one side, the backend code develop on Python Notebook (code_streamlit_np.ipynb) to find the best Hyperparameter and testing the code.
on the second there is a part developed in Streamlit for reproduction of testing result.

### Algorithm that i analize during the thesis:

**1)Logistic regression and Penalized logistic regression (ridge and lasso)**

**2)Lda**

**3)Random Forest**

**4)Neural Network**

the web application Streamlit, that is deployed on internet with Heroku, to allow to perform the same analysis without previous knowledge of python and coding:

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
