# Apply LSTM to predicting future trends of stock price
Taking 8 types of trading data of CSI300 Index constituent shares from Jan.1st 2006 to Jul.31st 2018 as samples, constructed RNN and LSTM models to predict average price changes/price fluctuation categories respectively; besides, standardized data with different ways, adjusted parameters, increased data by transformation, tried different ways to prevent over-fitting (dropout & L2 regularization) to optimize prediction accuracy. As a result, it found that prediction on price fluctuation categories performed much better, and the out-of-sample test accuracy reached 73%.  
### Step1: Collecting trading data      
Input data includes the open, high, low, close, volume, amount, percent change, turnover of all the constituent stocks of CSI300 index from Jan.1st 2006 to Jul.31st 2018.  
### Step2: Data preprocessing
•	Removed stocks that had been suspended for more than 90 days  
•	Increased data by transformation, final dataset dimension was 35500*30*8  
•	Normalized trading data  
•	Divided it into training & test set and set labels, shuffled it  
![image](https://github.com/Luffy-wu/picture/blob/master/图片%2081.png)        
![image](https://github.com/Luffy-wu/picture/blob/master/图片%2082.png)        
### Step3: Model iteration
##### •	Predict average price changes
At first, used trading data of last 30 days to predict average changes of price in next 3 days, and the output label was specific value. But, it was hard to give an accurate prediction.
![image](https://github.com/Luffy-wu/picture/blob/master/图片%2083.png)      
##### •	Predict price fluctuation categories
Then, categorized price changes into three types, used trading data of last 30 days to predict the type of average changes of price in next 3 days, and the output labels were one-hot variables representing “sharp rise, sharp decline, moderate fluctuation”, determined by 1/3 quantile (this meant that initial accuracy was 0.33). Finally, the out-of-sample test accuracy of RNN reached 69.4% and the accuracy of the optimal LSTM model reached 73%, performing slightly better than RNN.
![image](https://github.com/Luffy-wu/picture/blob/master/图片%2084.png)        
![image](https://github.com/Luffy-wu/picture/blob/master/图片%2085.png)      
