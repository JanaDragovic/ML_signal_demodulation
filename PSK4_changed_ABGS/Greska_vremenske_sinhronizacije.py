import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# (2) Slucaj izmenjene konstelacije QPSK simbola - analiza sa ABGS kanalom
# (1) Prvi način obrade


# Ucitavanje podataka
data_for_ml = scipy.io.loadmat('DataForML_PSK4_changed.mat')
data_classic_decision = scipy.io.loadmat('Data_Klasicno_odlucivanje_PSK4_changed.mat')
data_pes = scipy.io.loadmat('DataPes_PSK4_changed.mat')

# Prikaz ključeva kako bismo razumeli strukturu svakog fajla 
data_classic_decision.keys(), data_for_ml.keys(), data_pes.keys()

# Ekstrakcija varijabli za obucavanje ML modela iz DataForML.mat
labels_complex_symbols = data_for_ml['Labele_Kompleksni_simboli']
labels_symbol_numbers = data_for_ml['Labele_Redni_brojevi_simbola']
received_symbols_sample = data_for_ml['Simboli_na_prijemu_uzorak']

# Prikaz oblika ovih nizova kako bismo razumeli njihove dimenzije
labels_complex_symbols.shape, labels_symbol_numbers.shape, received_symbols_sample.shape

# Priprema podataka za obucavanje i testiranje
# S obzirom da je received_symbols_sample struktuiran kao 8 redova (nivoi SNR-a) po 65536 kolona (simboli),
# reshape-ovacemo i selektovati podatke u skladu s tim

# Reshape primljenih simbola da imaju nivoe SNR-a u odvojenim nizovima
received_symbols_reshaped = np.transpose(received_symbols_sample)

# Selektovanje trening i testing podataka
# Trening podaci: prvih 20000 simbola za svaki SNR
train_data = received_symbols_reshaped[:20000, :]
train_labels = labels_symbol_numbers[:20000]

# Verifikacioni podaci: sledecih 5000 simbola za svaki SNR (20000 do 25000)
verification_data = received_symbols_reshaped[20000:25000, :]
verification_labels = labels_symbol_numbers[20000:25000]

# Podaci za testiranje performansi: sledecih 25000 simbola za svaki SNR (25000 do 50000)
performance_test_data = received_symbols_reshaped[25000:50000, :]
performance_test_labels = labels_symbol_numbers[25000:50000]

train_data.shape, train_labels.shape, verification_data.shape, verification_labels.shape, performance_test_data.shape, performance_test_labels.shape


train_data_real_imag = np.hstack((train_data.real, train_data.imag))
verification_data_real_imag = np.hstack((verification_data.real, verification_data.imag))

# Spremanje labela (pretpostavka je da su već ravne, tj. 'flattened')
train_labels_flat = train_labels.ravel()
verification_labels_flat = verification_labels.ravel()

# Kreiranje i obučavanje SVM modela sa transformisanim podacima
svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_model.fit(train_data_real_imag, train_labels_flat)

# Verifikacija modela sa transformisanim podacima za verifikaciju
svm_verification_accuracy = svm_model.score(verification_data_real_imag, verification_labels_flat)

# Ispis tačnosti verifikacije
print("Tačnost verifikacije SVM modela:", svm_verification_accuracy)


# Obučavanje Naïve Bayes modela sa transformisanim podacima
nb_model = GaussianNB()
nb_model.fit(train_data_real_imag, train_labels_flat)

# Verifikacija Naïve Bayes modela sa transformisanim podacima za verifikaciju
nb_verification_accuracy = nb_model.score(verification_data_real_imag, verification_labels_flat)

# Ispis tačnosti verifikacije za Naïve Bayes model
print("Tačnost verifikacije Naïve Bayes modela:", nb_verification_accuracy)


# Predikcije modela za test set
svm_predictions = svm_model.predict(verification_data_real_imag)
nb_predictions = nb_model.predict(verification_data_real_imag)

# Matrica konfuzije za SVM model
svm_confusion_matrix = confusion_matrix(verification_labels_flat, svm_predictions)
print("Matrica konfuzije za SVM model:")
print(svm_confusion_matrix)

# Matrica konfuzije za Naïve Bayes model
nb_confusion_matrix = confusion_matrix(verification_labels_flat, nb_predictions)
print("Matrica konfuzije za Naïve Bayes model:")
print(nb_confusion_matrix)




