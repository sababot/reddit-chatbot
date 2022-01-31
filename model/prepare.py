import re

questions = []
answers = []

with open('questions.txt') as f1:
	for text in f1:
		questions.append(str(text))

with open('answers.txt') as f2:
	for text in f2:
		answers.append(str(text))