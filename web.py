import streamlit as st
import numpy as np
import requests
from tools import draw_gauge
if 'ID'  not in st.session_state.keys():
    st.session_state['ID'] = None
def get_id():
    return st.session_state['ID'] or None


if __name__ == "__main__":

    st.title('Implementez un modèle de scoring')
    if get_id() is None:
        data = requests.get(f"http://localhost:3000/load_initial_data/v2")
        ids = data.json() and data.json()['ids'] or False
        values = data.json() and data.json()['values'] or False
        print('Intial session state:',list( ids.values())[0] )
        st.session_state['ID'] =list(ids.values())[0]

    else:
        data = requests.get(f"http://localhost:3000/load_data/v2/"+ str(get_id()))
        ids = data.json() and data.json()['ids'] or False
        values = data.json() and data.json()['values'] or False

    st.subheader("Select the client ID")

    # with st.expander("See full container of datatable"):
    #     st.write(df.columns)
    st.write("Choose sk_id")
    if ids:
        sk_id = st.selectbox("Id", options=list(ids.values()), on_change=get_id, index=0, key="ID")

    with st.form("form_key"):
        col1, col2 = st.columns(2)

        with col1:
            if values:
                code_gender = values["code_gender"]
                occupation_type = values['occupation_type']
                income_type = values['name_income_type']
                education_type = values['education_type']
                housing_type = values['housing_type']

                st.write(f'Gender: {code_gender}')
                st.write(f'Occupation_type: {occupation_type}')
                st.write(f'Income_type: {income_type}')
                st.write(f'education_type: {education_type}')
                st.write(f'housing_type: {housing_type}')

            else:
                st.warning("No data found for the selected SK_ID.")

        with col2:

            if values:
                amt_credit = values["amt_credit"]
                amt_income_total = values["amt_income_total"]
                amt_annuity = values["amt_annuity"]
                days_employed = values["days_employed"]
                old = values["days_birth"]
                st.write(f'Amount_Credit: {amt_credit}')
                st.write(f'Amount_Annuity: {amt_annuity}')
                st.write(f'Total_Income_Amount: {amt_income_total}')
                st.write(f'Days_employed: {np.round(days_employed/365, 0)} years')
                st.write(f'age_of_client: {np.round(-old/365, 0)} years')

            else:
                st.warning("No data found for the selected SK_ID.")

        st.divider()

        tab1, tab2, tab3 = st.tabs(['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY'])

        with tab1:
            st.image('AMT_CREDIT.png',caption='amt_credit')

        with tab2:

            # Plot the distribution of ages in years
            st.image('AMT_INCOME_TOTAL.png', caption='amt_income_total')

        with tab3:

            # Plot the distribution of ages in years
            st.image('AMT_ANNUITY.png', caption='amt_annuity')

        st.divider()

        tab11, tab12 = st.tabs(['Employment', 'Age of client'])

        with tab11:
            col = "DAYS_EMPLOYED"
            st.image('DAYS_EMPLOYED.png', caption='days_employed')
        with tab12:
            col = "DAYS_BIRTH"
            st.image('DAYS_BIRTH.png', caption='days_birth')
        submit_btn = st.form_submit_button(label='Submit', type='secondary')

        if submit_btn:
            # st.write(pd.DataFrame(values))

            req = requests.get(f"http://localhost:3000/predict?id={sk_id}")
            req_json = req.json()
            proba = np.round(req_json["proba"][0][1], decimals=3)

            # Exemple d'utilisation
            max_value = 100
            current_value = np.round(proba*100, 2)
            st.write(f"Valeur actuelle du niveau de risque : {current_value}/{max_value}")
            draw_gauge(current_value, max_value)
