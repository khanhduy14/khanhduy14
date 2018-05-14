from underthesea import pos_tag
from underthesea import word_tokenize
import pycrfsuite

def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]


    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]


    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        features.append('BOS')


    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')
    return features



def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

def predict(sentence):
    sentences = word_tokenize(sentence)
    tagger = pycrfsuite.Tagger()
    tagger.open('crf.model')
    tokens = [(word, tag, "X") for word,tag in pos_tag(sentence)]
    X = extract_features(tokens)
    y_pred = tagger.tag(X)
    output = [(tag, token) for token, tag in zip(y_pred, sentences)]
    return output


print(predict("Quán nhỏ nhưng có võ, vị chi cũng hơn 15 món đặc sản miền trung: lòng xào nghệ, hến xào, ốc hút..., chỗ này rẻ, ăn no nê tầm 70k mỗi người là phê. "))
