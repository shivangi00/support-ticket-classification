# support-ticket-classification

This project chooses a different path for each pipeline after shared preprocessing shaped the data. One relied on Porter Stemming plus TF-IDF weighted bigrams for its structure. The second turned to WordNet Lemmatisation along with distributed representations from Word2Vec. Ticket routing covered 14 distinct labels through both routes. Abbreviations were expanded early, noise was reduced without losing negations, and terms were standardised before analysis began. A common framework allowed fair measurement despite differing methods. Classification used Gaussian Naive Bayes, so the model differences highlighted feature effects clearly.

What stands out most in the numbers is how Pipeline 2 does better than Pipeline 1 on every measure - Macro F1 sits at 0.7772 compared to 0.6731. Looking closer at individual classes, one thing becomes clear: Pipeline 1 collapses entirely on account_recovery, scoring zero, whereas Pipeline 2 reaches 0.75, hinting that meaning-rich embeddings grasp patterns missed by rigid TF-IDF vectors. From examining
actual predictions, many mistakes appear tied less to flaws in design and more to unclear boundaries within the label system itself. Despite higher scores, confusion still arises where categories overlap in subtle ways.

| Step  | Pipeline 1 |  Pipeline 2    |
|-------|-----|-------|
| Normalisation | Domain abbreviation expansion (27 patterns)  | Same (shared) |
| Tokenisation   | Regex: re.findall([a-z0-9]+)  | Same (shared)   |
| Stop Words   | NLTK English; negations retained  | Same (shared)   |
| Morphological   | Stemming (PorterStemmer)  | Lemmatisation (WordNetLemmatizer)   |
| Representation   | TF-IDF, ngram_range=(1,2)  | Word2Vec CBOW, vector_size=100, window=1   |
| Classifier   | Gaussian Naive Bayes  | Gaussian Naive Bayes   |
