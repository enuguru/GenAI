from summarizer import Summarizer
text = open("article.txt").read()
model = Summarizer('distilbert-base-uncased')
result = model(text, num_sentences=3)
print(result)
