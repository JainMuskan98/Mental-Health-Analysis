import streamlit as st
import pickle
from PIL import Image
import numpy as np
model=pickle.load(open('model.pkl','rb'))



def predict_forest(Sadness,Loneliness,Overthinking):
    input=np.array([[Sadness,Loneliness,Overthinking]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    image = Image.open('download.jpg')

    st.image(image, caption='',width = 700, use_column_width= True)
    # st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.title("The Machine Learning Project")
    html_temp = """
    <div style="background-color:#3ebdaa ;padding:10px">
    <h2 style="color:white;text-align:center;">Mental Health Analysis </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Sadness = st.slider("How sad are you?", 1, 5)
    Loneliness = st.slider("Are you lonely?", 1, 5)
    Overthinking = st.slider("How much do you overthink", 1, 5)
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> KEEP IT UP :)</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> ITS TIME TO SPEAK UP!!! </h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_forest(Sadness,Loneliness,Overthinking)
        st.success('The probability that your mental health is safe is {}'.format(output))

        if output > 0.5:
            st.markdown(safe_html,unsafe_allow_html=True)
        else:
            st.markdown(danger_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()