# To build Machine Learning based web application
from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap

# For Text and Word classification
from textblob import TextBlob, Word

# for random stuff generation and prediction in NLP
import random

# To record the execution time
import time

# by this app understand that program will run on web browser
app = Flask(__name__)
Bootstrap(app)
app.config['DEBUG'] = True

@app.route("/", methods = ['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/analyse", methods = ["POST"])
def analyse():

    # recording execution time
    startTime = time.time()

    # request method must be POST
    if request.method == 'POST':

        # raw text data coming from textarea
        rawtext = request.form['rawtext']

        # passing raw text data to Textblob
        blob = TextBlob(rawtext)

        # saving blob to another variable
        receivedText = blob

        # extracting polarity and subjectivity from blob
        blobPolarity, blobSubjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity

        # count of words
        numberOfTokens = len(list(blob.words))
        
        # nouns will be used in text categorization
        nouns = list()
        for word, tag in blob.tags:
            
            # 'NN' = default value of every tag in textblob
            if tag == 'NN':
            
                # lemmatize the nouns
                nouns.append(word.lemmatize())
                lenOfWords = len(nouns)

                # words which are not nouns will be categorised as random words
                randomWords = random.sample(nouns, len(nouns))
                finalWord = list()
            
                for item in randomWords:
            
                    # pluralize means what the text is talking about
                    word = Word(item).pluralize()
                    finalWord.append(word)
                    summary = finalWord

                    # end of execution time recording 
                    endTime = time.time()
                    finalTime = endTime - startTime
    
    return render_template('index.html', receivedText=receivedText, numberOfTokens=numberOfTokens, lenOfWords=lenOfWords, \
        blobPolarity=blobPolarity, blobSubjectivity=blobSubjectivity, summary=summary, finalTime=finalTime)


if __name__ == "__main__":
    app.run()