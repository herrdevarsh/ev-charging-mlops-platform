import streamlit as st

from ev_charging_mlops_platform.predict import ModelService


@st.cache_resource
def get_model_service() -> ModelService:
    return ModelService()


def main():
    st.title("EV Charging Load Predictor")

    st.markdown("Predict estimated **sessions per day** for a charging station.")

    col1, col2 = st.columns(2)

    with col1:
        region = st.text_input("Region (State/Province)", "Berlin")
        city_type = st.selectbox("City Type", ["urban", "suburban", "rural"])
        charger_type = st.selectbox("Charger Type", ["AC", "DC"])

    with col2:
        power_kW = st.number_input("Max Power (kW)", min_value=1.0, value=150.0, step=1.0)
        num_connectors = st.number_input("Number of connectors", min_value=1, value=4, step=1)

    if st.button("Predict"):
        payload = {
            "region": region,
            "city_type": city_type,
            "charger_type": charger_type,
            "power_kW": power_kW,
            "num_connectors": num_connectors,
        }

        try:
            model_service = get_model_service()
            pred = model_service.predict(payload)
            st.success(f"Estimated sessions per day: **{pred:.1f}**")
            st.caption("Note: based on Open Charge Map location data + proxy utilization label.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error during prediction: {exc}")


if __name__ == "__main__":
    main()
