import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

training_sentences = [

    "what are your working hours",
    "when are you open",
    "opening time",
    "closing time",
    "what time do you close",
    "are you open today",
    "working schedule",
    "business hours",

    "how can i book an appointment",
    "i want to see a doctor",
    "schedule a visit",
    "book a visit",
    "doctor appointment",
    "make an appointment",
    "how to register for a visit",
    "appointment booking",

    "this is an emergency",
    "i need urgent help",
    "emergency help",
    "urgent medical issue",
    "serious condition",
    "i need immediate assistance",
    "medical emergency",
    "urgent situation",

    "do you offer covid testing",
    "covid test",
    "coronavirus testing",
    "covid examination",
    "covid check",
    "pcr test",
    "rapid covid test",
    "covid symptoms test",

    "do you accept insurance",
    "health insurance",
    "can i use my insurance",
    "insurance coverage",
    "insurance accepted",
    "which insurance do you accept",
    "insurance providers",
    "insurance support",

    "where are you located",
    "clinic address",
    "how to find you",
    "location of the healthcare center",
    "your address",
    "directions to clinic",

    "what doctors work here",
    "available doctors",
    "specialists",
    "do you have a cardiologist",
    "doctor list",
    "medical staff",

    "what services do you provide",
    "medical services",
    "available treatments",
    "healthcare services",
    "what can you help with",
    "services list",

    "how can i pay",
    "payment methods",
    "do you accept cash",
    "card payment",
    "payment options",
    "how do i pay for services",

    "how to cancel an appointment",
    "cancel my visit",
    "appointment cancellation",
    "i want to cancel appointment",
    "cancel booking",
    "how to reschedule"
]

training_labels = (
    ["opening_hours"] * 8 +
    ["appointment"] * 8 +
    ["emergency"] * 8 +
    ["covid"] * 8 +
    ["insurance"] * 8 +
    ["location"] * 6 +
    ["doctors"] * 6 +
    ["services"] * 6 +
    ["payment"] * 6 +
    ["cancel"] * 6
)

training_sentences = [clean_text(s) for s in training_sentences]

responses = {

    "opening_hours":
    "🕒 **Working Hours**\n\n"
    "Our healthcare center is open **Monday to Friday, 8:00 AM – 6:00 PM**.\n"
    "We are closed on weekends and public holidays.",

    "appointment":
    "📅 **Appointment Booking**\n\n"
    "You can book an appointment by:\n"
    "• Calling our reception desk\n"
    "• Using our online booking system\n"
    "• Visiting the clinic in person\n\n"
    "Please arrive **10 minutes early**.",

    "emergency":
    "🚑 **Emergency Assistance**\n\n"
    "If you are facing a medical emergency:\n"
    "• Call emergency services immediately\n"
    "• Go to the nearest emergency department\n\n"
    "Online chat is NOT suitable for emergencies.",

    "covid":
    "🦠 **COVID-19 Testing**\n\n"
    "We provide:\n"
    "• PCR tests\n"
    "• Rapid antigen tests\n\n"
    "Testing is available **by appointment only**.",

    "insurance":
    "💳 **Insurance Information**\n\n"
    "We accept most major health insurance providers.\n"
    "Please bring your insurance card and ID.",

    "location":
    "📍 **Our Location**\n\n"
    "The healthcare center is located in the city center.\n"
    "Public transport and parking are available nearby.",

    "doctors":
    "👩‍⚕️ **Medical Staff**\n\n"
    "Our specialists include:\n"
    "• General practitioners\n"
    "• Cardiologists\n"
    "• Dermatologists\n"
    "• Pediatricians",

    "services":
    "🩺 **Healthcare Services**\n\n"
    "We offer:\n"
    "• General consultations\n"
    "• Specialist visits\n"
    "• Laboratory tests\n"
    "• Preventive healthcare",

    "payment":
    "💰 **Payment Methods**\n\n"
    "We accept:\n"
    "• Cash\n"
    "• Credit / debit cards\n"
    "• Insurance-covered payments",

    "cancel":
    "❌ **Appointment Cancellation**\n\n"
    "You can cancel or reschedule your appointment:\n"
    "• By phone\n"
    "• Through the online system\n\n"
    "Please notify us **24 hours in advance**."
}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = LogisticRegression(max_iter=1000)
model.fit(X, training_labels)

def chatbot_response(user_input):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    intent = model.predict(vector)[0]
    return responses[intent]

st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥")
st.title("🏥 Healthcare Center Chatbot")
st.write("Ask about appointments, working hours, emergencies, COVID tests, insurance and more.")

user_input = st.text_input("💬 Your question:")

if user_input:
    st.markdown(f"**🤖 Bot:**\n\n{chatbot_response(user_input)}")
