from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding , Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D 
from tensorflow.keras.optimizers import Adam , RMSprop
from tensorflow.keras.layers import Bidirectional , Dense 
from tensorflow.keras.layers import LSTM , GRU
import numpy as np
import matplotlib.pyplot as plt 


data="Maodo\nIl rassemble en lui toutes les Vertus\nIl atteste une Hauteur qui ne saurait se rencontrer que chez les Meilleurs\nIl est dans la Voie de la Maîtrise\nIl subvient à toute Demande tout en gardant le visage Gracieux et Bien Venant\n Il ne donne pas peu il est abondant\nIl est Noble dans ses Actes il ressemble à la Pleine Lune\n Noble Et Pure\n Il est le Serviteur de son Hôte\n Dans son Hospitalité il est un Homme Libre qui ne dispose pas de lui même\n Ce qu'il donne reste à jamais\nLa Patience est la Fraîcheur de son Parfum\nAu Moment où les Hommes polémiquent et parlent de futilités\n Son ouïe rejette toute Obscénité il est sourd à l'Insolence\nIl n'a pas laissé manifester de Colère que celle-ci soit imprégnée de Miséricorde\nNi d'Hostilité que celle ci ne le soit d'Amour\n Son Âme Noble aspire au Zikr de la Salatul Fatihi\n Il est dans la Liberté la Maîtrise et la Perfection\nIl ne parle jamais de ses Besoins sa Pudeur lui suffit\n Il suit les Habitudes de son Ami Intime\n Son Intimité n'est légitime qu'avec Son Bien Aimé saw\n Il se revêt de tous ses Habits\n Il porte à chaque étape le Vêtement Approprié\nIl occupe la Place d'Honneur\n Avec des Qualités de Grandeurs et de Splendeurs\n Il possède le Manteau qui englobe les Deux Dimensions du Zahir et du Batin\n Ces Belle Vertus représentent une Belle Parure\n Il se fait Beau parce qu'Allah est beau et aime la Beauté\n Il est revêtu de l'état Particulier du Seydi\nIl est avant Tout un Homme Généreux\n Il n'exclut personne\n il comble de ses Dons\nIl donne sans compter comme la Rosée du Petit Matin\n Comme la Pluie qui secourt il ne trouve le Repos que dans le Don\nOn peut compter sur lui\n à Tout Moment\n Matin et Soir\n Aussi souvent que l'on veut\nOn sait qu'en toutes ces Circonstances il sera toujours aussi Généreux\n Il est dans la Constance de la Générosité avec une telle Abnégation\nIl sacrifie sa Part et la donne au disciple\nIl supporte Tout sans se plaindre\nSa Largesse de Cœur supposent Trois Qualités Fondamentales\nLa Patience le Pardon et la Longanimité\n Cette Patience qui implique l'Endurance et la Maîtrise de Soi\nLes Nobles Vertus et l'Amour Muhammadien  sont ainsi son Héritage et également un Dépôt\n Il possède cette Connaissance qui lui permet de savoir comment\n Quand et avec qui user de ces Noble Vertus\n Il est dans la Seigneurie par Excellence\nIl est d'une Science Abondante\nSa Science représente la Science du Décret et de l'Ordre Divin\nLa Science des Mesures de l'Univers et celle de la Présence Divine\nIl traite chaque chose selon sa Mesure\n  Il est le Singulier il est l'Unique\n Il est l'Unique parmi ses Semblables\n Il est dans une Générosité et une Indulgence Infinies\n Il garde l'équilibre entres des vertus Contraires\n Et adopte un Comportement à chaque Situation\n Douceur ou Fermeté\n Rigueur ou Générosité\n Il est l'Archétype du Chevalier Tidjani au Service des Créatures\n Son Abnégation est sa Quintessence\nIl préfère les autres à lui même\nIl préfère la Umma à sa Propre Nafs\nIl dispose de la Vigueur qui lui permet de combattre son ego \n De la Générosité et de l'Abnégation qui lui font préférer l'ordre d'Allah à ses Propres Désirs\n Il est dans la Soumission Absolue à l'ordre Divin\n Il est le Gardien des Trésors\n Ces Trésors qui contiennent les Moyens du Bonheur des Hommes\n Il est le Trésor \n le Coffre Fort et le Gardien\n Il est le Maître Serviteur\n celui qui aatteint le Sommet \n Un Homme Inconnu parmi le Commun des Hommes"

corpus = data.lower().split("\n")

tokenizer=Tokenizer()
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences (corpus)

total_words = len(word_index) +1


input_sequences =[]

for line in corpus :
    list_token=tokenizer.texts_to_sequences([line])[0]
    for i in range (1,len(word_index)):
        n_gram_sequences = list_token[:i+1]
        input_sequences.append(n_gram_sequences)

max_sequences_len = max([len(x) for x in input_sequences])
input_sequences = np.array([pad_sequences(input_sequences , maxlen = max_sequences_len , padding='pre')])


xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequences_len-1))
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy' , optimizer='adam' , metrics=['accuracy'])
history=model.fit(xs , ys , epochs=500 , verbose =1)


def graph_plot(history , string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string])
    plt.show()

graph_plot(history , 'accuracy')


#Here we predict new words 












