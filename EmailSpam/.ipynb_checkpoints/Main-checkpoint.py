import imaplib
import joblib
from email.parser import BytesParser
from email import policy
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained Naive Bayes classifier and CountVectorizer
nb_classifier = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')


def predict_spam(input_text):
    preprocessed_text = preprocess(input_text)
    input_vectorized = vectorizer.transform([preprocessed_text])
    prediction = nb_classifier.predict(input_vectorized)
    if prediction[0] == 1:
        print("The input text is classified as spam.")
    else:
        print("The input text is not classified as spam.")


def preprocess(text):
    preprocessed_text = text.lower()
    return preprocessed_text


# def check_latest_email():
#     mail = imaplib.IMAP4_SSL('imap.gmail.com')
#     mail.login('your_email@gmail.com', 'your_password')
#     mail.select('inbox')
#     result, data = mail.search(None, 'ALL')
#     latest_email_id = data[0].split()[-1]
#     result, data = mail.fetch(latest_email_id, '(RFC822)')
#     raw_email = data[0][1]
#     msg = BytesParser(policy=policy.default).parsebytes(raw_email)
#     subject = msg['subject']
#     body = msg.get_body(preferencelist=('plain')).get_content()

#     # Close the connection to the email server
#     mail.logout()

#     return subject, body

# Check the latest email 
# subject, body = check_latest_email()
# print("Subject:", subject)
# print("Body:", body)
# input_text = subject + " " + body
input_text = "Congratulations! You've won a free vacation. Click here to claim your prize."
predict_spam(input_text)
