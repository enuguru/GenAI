from transformers import pipeline
text = open("article.txt").read()
question = "What is BOE doing?"

answering = pipeline("question-answering",
                     model='distilbert-base-uncased-distilled-squad')
result = answering(question=question, context=text)
print(result)
