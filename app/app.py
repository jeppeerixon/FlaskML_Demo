from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid 


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hejhej():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/smalm2.jpg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "app/static/" + random_string + ".svg"
        np_arr = float_string_to_np_arr(text) 
        new_model = load('app/model2.joblib')
        make_pic('app/sm_lgh_pris.csv', new_model, np_arr, path)
        return render_template('index.html', href=path[4:])


def make_pic(traning_data_filename, model, new_inp_np_arr, output_file):
  data = pd.read_csv(traning_data_filename, sep=';')
  kvm = data["Kvm"]
  pris = data["Price"]
  data = data[kvm < 55]
  kvm = data["Kvm"]
  pris = data["Price"]
  pris_np = data['Price']
  x_new = np.array(list(range(55))).reshape(55, 1)
  preds = model.predict(x_new)

  fig = px.scatter(x=kvm, y=pris, title="Slutpris lägenheter södermalm", labels={'x': 'Kvm', 'y': 'Pris'})
  fig.add_trace(go.Scatter(x=x_new.reshape(55), y=preds, mode='lines', name='Model'))

  new_preds = model.predict(new_inp_np_arr)

  fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Output', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
  
  fig.write_image(output_file, width=800, engine='kaleido')
  fig.show()


def float_string_to_np_arr(float_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in float_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)


#.env\Scripts\activate.bat
#aktivera venv om det kopplats borts

#1. skapa env
#2. ladda ner och rensa data
#3. skapa en graf från datan
#4. gör en LR modell utifrån datan
#5. ladda ned modell och testa
#6. Gör funktion för input av data + cleana datan
#7. Funktion för skapa bild av nya datan i modellen
#8. implementera kod i Flask
#9. hemsidan visas via index.html
#10. funktionerna körs via app.py genom Get/Post methods
#11. import uuid skapar random namn som skapar nytt bild hela tiden för att undvika cache
#12. Verkar strula lite med scikit learn versioner på datan/local env.
