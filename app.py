from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]


    stop_words = set(stopwords.words("english"))
    words = [[word for word in sentence if word not in stop_words] for sentence in words]

    return words

# Create the response function
# def generate_response(user_input, sentences, tfidf_vectorizer, tfidf_matrix):
#     user_input = preprocess_text(user_input)
#     user_input = [' '.join(words) for words in user_input]

#     response_index = cosine_similarity(tfidf_vectorizer.transform(user_input), tfidf_matrix).argmax()
#     return sentences[response_index]
def generate_response(user_input, sentences, tfidf_vectorizer, tfidf_matrix, num_responses=3):
    user_input = preprocess_text(user_input)
    user_input = [' '.join(words) for words in user_input]

    response_indices = cosine_similarity(tfidf_vectorizer.transform(user_input), tfidf_matrix).argsort()[0][-num_responses:][::-1]
    responses = [sentences[i] for i in response_indices]

    return '\n'.join(responses)

document_text = """<APPLICATIONS OF NANO TECHNOLOGY 
Thus, at nano scale, optical, electrical, magnetic, mechanical properties of the materials 
change. These nano materials, having new properties can be used in variety of applications in 
different fields like – food processing, medicine, automobiles, paint technology, computer 
technology, robotics, space technology etc. Advances in nanotechnology have made it 
possible to build devices and machines like nano-assembler which assembles the molecules at 
atomic level very fast. 
1) Medicine 
Nano particles can be used for detection and treatment of cancers and tumours. 
Researchers are developing customized nano particles which can be injected into the 
body. The drugs can be encapsulated in nano particles that can deliver drugs directly 
to diseased cells in the body.  When it's perfected, this method should greatly reduce 
the damage treatment such as chemotherapy does to a patient's healthy cells. 
Nanotechnology tests are being developed for fast detection of viruses and antibodies. 
2) Electronics 
3) Energy 
The semiconductor devices work on the concept of charge transport only. Nano sized 
electronic components work on the concept of charge transport as well as spin 
transport of electrons. Using nano sized components reduce power consumption while 
decreasing weight and size of the device. This makes us possible to have more density 
of components and lead to smaller and faster processors. Nano devices have increased 
the data storage capacities of the memory devices. 
Nanotechnology is being used to reduce the cost of catalysts used in fuel cells to 
produce hydrogen ions from fuel such as methanol and to improve the efficiency of 
membranes used in fuel cells to separate hydrogen ions from other gases such as 
oxygen. Researchers are working on the idea of trapping and storing hydrogen using 
carbon nano tubes. Attempts are being made to develop nanotech solar cells that can 
be manufactured at significantly lower. Nano materials are also used to increase 
energy density of rechargeable batteries which are used in laptops and mobile phones. 
4) Space Technology 
With advancements in nano materials it is possible to develop lightweight spacecraft 
and a cable for the space elevator. By significantly reducing the amount of rocket fuel 
required, these advances could lower the cost of reaching orbit and travelling in space. 
Polymer composites using silica fibres and nano materials have larger mechanical 
strength and low temperature coefficient of expansion. The spacecrafts made with 
such materials can withstand high temperature and conditions during launching and 
re-entry into the earth’s atmosphere. Increase in efficiency of solar cells using 
nanotechnology made it possible to enhance the energy supply to the spacecrafts and 
satellites. 
5) Automobiles 
Nano tube composite have better mechanical strength compared to steel. Efforts are 
being made to develop cheaper nano tube composites that can replace steel which is 
used to construct the body structure of automobiles. Use of nano particles in paints 
provides thin and smooth coatings.  
Nano particle catalysts can be used to reduce the temperature required to convert raw 
materials into fuel or increase the percentage of fuel burned at a given temperature. 
Catalysts made from nano particles have a greater surface area to interact with the 
reacting chemicals than catalysts made from larger particles. The larger surface area 
allows more chemicals to interact with the catalyst simultaneously, which makes the 
catalyst more effective. This increased effectiveness can make a process such as the 
production of diesel fuel from coal more economical, and enable the production of 
fuel from currently unusable raw materials such as low grade crude oil. 
6) Environmental 
Nano particle based sensors are capable of detecting water and air pollution due to 
toxic ions and pesticides with a very high sensitivity. Nano material catalysts can be 
used to convert the harmful emissions from industries and automobiles to less harmful 
gases. 
7) Textiles 
The use of nanotechnology in textile industry has made it possible to fabricate water 
repellent and wrinkle free clothes.>"""


processed_sentences = preprocess_text(document_text)
sentences = [" ".join(words) for words in processed_sentences]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = generate_response(user_input, sentences, tfidf_vectorizer, tfidf_matrix)
    return {"response": response}

if __name__ == "__main__":
    app.run(debug=True)
