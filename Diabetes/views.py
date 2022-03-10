from django.shortcuts import render
import pickle


def home(request):
    return render(request, 'index.htm')


def prediction(pregnancies, glucouse, bp, skinthickness, insulin, bmi, dpf, age):
    model = pickle.load(open('diabetes_rfc.pkl', 'rb+'))
    scaler = pickle.load(open("scaler.sav", 'rb'))
    inputs = [[pregnancies, glucouse, bp,
               skinthickness, insulin, bmi, dpf, age]]
    pred = model.predict(scaler.transform(inputs))
    proba = model.predict_proba(scaler.transform(inputs))
    return pred, proba


def submit(request):
    pregnancies = int(request.POST['Pregnancies'])
    glucouse = int(request.POST['Glucose'])
    bp = int(request.POST['BloodPressure'])
    skinthickness = int(request.POST['SkinThickness'])
    insulin = int(request.POST['Insulin'])
    bmi = float(request.POST['BMI'])
    dpf = float(request.POST['DiabetesPedigreeFunction'])
    age = int(request.POST['Age'])

    predicted, probability = prediction(
        pregnancies, glucouse, bp, skinthickness, insulin, bmi, dpf, age)
    probability = round(probability[0][predicted][0], 3)*100
    return render(request, 'index.htm', {'Predicted': predicted, "Probability": probability})