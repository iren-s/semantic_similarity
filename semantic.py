import spacy
# Load the medium-sized English model we are going to use
nlp = spacy.load('en_core_web_md')

#Example 1 from PDF
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))


# Example2 from PDF
tokens = nlp("cat apple monkey banana")
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Example 3 from PDF
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, where is my car",
"I\'ve lost my car in my car", 
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + "-", similarity)


"""
NOTE ON SIMILARITIES BETWEEN CAT, MONKEY AND BANANA
As we can see from the output, cat and monkey have the highest similarity which is predictable 
as they are both animals. The score between monkey and banana is much lower, but higher than between banana and cat,
as we assosiate monkey with banana. And finally, the lowest similarity score between cat and banana, as expected, 
because they don't have much in common. 
"""

# My own example where three words are checked for similarities
word1 = nlp("men")
word2 = nlp("women")
word3 = nlp("car")

print(word1.similarity(word2))
print(word1.similarity(word3))
print(word2.similarity(word3)) 
"""
In this example we can see that men and woman have a high similarity score of 0.776, which make sense since
they are both referring to people's gender.
The similarity beetween man/women and car is very low as they are not related concepts (0.088 for man-car pair).
Interestingly enough, the score of woman-car pair is negative (-0.029), which means that the words are very dissimilar
in terms of their meanings.
"""

"""
After running the same examples through "en_core_web_sm" model I have made following observations.
Example with cat, monkey and banana shows high similarity score for all words combinations with the highest score 
between monkey and banana (0.72), when with model "en_core_web_sm" the score was much lower (0.40).
in my example "en_core_web_md" model produced lower similarity scores than the other model. However, the relative
differences between scores were consistent. Man and woman showed higher similarity between each other in both models
and similarities between man and car the same as between woman and car were much lower.
"""
