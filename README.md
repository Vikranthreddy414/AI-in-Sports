# Cricket Commentary Sentiment Analysis and Generation
- NLP Project which aims to dissect cricket commentary data, find the sentiment of cricket commentary and aim to generate cricket commentary automatically given event generated through a prompt
- We have Asia cup data which is smaller which we use for sentiment analysis and we have IPL ball by ball data for 3 years which is bigger for training a model for automatic commentary generation
- All data is from Kaggle
- Sentiment analysis is performed to see if commentary given is neutral in sentiment or slightly positive
- Up to now websites like cricbuzz provide commentary for matches by manual typing. I aim to automate this process with a commentary generator given keywords of events obtained by some manner (eg image processing)

# Algorithms used

### Sentiment Analysis
We try out and compare 2 algorithms
- VADER (Valence Aware Dictionary and Sentiment Reasoner) lexicon and rule based bag of words model
- Roberta Pre-trained model (transform based)

### Commenrary Generation
- GPT-2 (Generative Pretrained Transform)

## Sentiment Analysis (Cricket_Commentary_Sentiment_Analysis.ipynb)
- Dataset link : https://www.kaggle.com/datasets/balabaskar/asia-cup-2022-ball-by-ball-data-and-commentary
- I first load and learn about the dataset
- I reduce the dimensionality of the dataset by picking only the required features
- I then pre-process by performing spelling correction, stopwords removal, tokenization and text encoding.
- I then apply both algorithms and obtain the polarity/sentiment scores
- Comparison of the models is done by a plot and conclusions are drawn (in depth in the notebook)

## Commentary Generation
 - Commentary Generator is done in google colab platform as we require google drive for storage and GPU for fast model training
 - Beware training a model using GPU may be expensive and may exhaust a free GPU not allowing you to use a GPU further for a couple of days in google colab
 
 We perform 3 steps to produce a model for automatic commentary generation
 1. Dataset preparation and pre-processing
 2. Model training
 3. UI creation to display model to the user
 
### Data pre-processing (CommentaryGeneration_DatasetPreperation.ipynb)

- Datasets link: https://www.kaggle.com/datasets/mhemendra/cricinfo-ipl-commentary
- Datasets are stored in drive and is mounted to colab
- Datasets are loaded and manipulated, columns are renamed and bowler/batsman names are extracted and put into seperate columns
- Only required rows are retained which reduces dimensionality and finally datasets are combined
- The batsman and bowler names are nouns which our model should not learn hence they are identified in the commentary text and substituted by generic terms like batsman and bowler so generated commentary is generic and does not suprise you with a pre learned cricketers name
- Spelling correction is done then data is split to train and test
This split data is then manipulated to the form which is accepted by GPT-2 algorithm and the data is now ready
- The pre-processed data can be found in the Data folder of the Sentiment Analysis section in this repository

### Model training (CommentaryGeneration_ModelTraining.ipynb)
- Using simpletransformers library we can quickly load a pre-trained GPT-2 language modelling model from huggingface
- We provide our custom arguments for the model like number of epochs (100), when to save the model (every 25 epochs) training batch size (10) and much more
- We then train and finetune the loaded pre-trained GPT-2 model using the training and testing data that was prepared in the previous step 
- Remember to use a GPU runtime for model training as it would take very long otherwise
- The best model is saved in the directory you specify
- I have already trained a model for 100 epochs if you do not want to spend the long time of training the model and would wish to load it and use it 
- Trained model drive link : https://drive.google.com/drive/folders/1ZAjNiva0JQfdVt6rG7gsmETgQiJl9Sfi?usp=sharing

### UI Creation and Model output (CommentaryGenerator_Output&UI.ipynb)
- Using gradio library we can create a simple yet elegent UI for our commentary generator easily
- We load our trained model and provide it arguments like length of commentary to be produced
- We use the generate function to generate commentary given the event that has occurred (SUCCESS we have a commentary generator)
- Output collected is manipulated slightly according to the format of websites and also such that only relevent results are displayed (Unwanted bits or irrelevent commentary is cut out)
- The UI is created so users can enter a batter, bowler name followed by an event and by a click of a button commentary is generated aptly and is displayed on the screen


# How the commentary generator looks like
![alt text](https://github.com/mrsurya1304/CricketCommentaryGenerator/blob/main/Samples/sample1.png)
![alt text](https://github.com/mrsurya1304/CricketCommentaryGenerator/blob/main/Samples/sample2.png)
![alt text](https://github.com/mrsurya1304/CricketCommentaryGenerator/blob/main/Samples/sample3.png)
![alt text](https://github.com/mrsurya1304/CricketCommentaryGenerator/blob/main/Samples/sample4.png)
![alt text](https://github.com/mrsurya1304/CricketCommentaryGenerator/blob/main/Samples/sample5.png)
 
# Conclusion
- Regarding sentiment analysis the Roberta model is more sure of the sentiment that it predicts than the VADER model
- Commentary is usually neutral but slightly skewed on the positive side and negative commentary is very low.
- The commentary generator works great and provides apt commentary for a wide range of events in cricket
- It is also able to give unique commentary each time for the same event entered
- There are some issues regarding nouns like cricketers name suddenly popping out of nowhere in the generated commentary. The reason for this is during training it has encountered some fielders names and names in the middle of the commentary text which I have not been able to extract out or replace
- Better performance can be achieved by fully eliminating nouns in the training data
- Commentary generated is relevent more times than not
- The notebooks are self explanatory and if you are lost there are comments to guide you along the way

