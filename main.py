import streamlit as st
import os
import shutil
from finalpredicted import predict_deepfake

def main():
    st.title("deepfake checker")
    st.write("This app detects deepfakes in videos.")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    method_mapping = {
        "MTCNN": "plain_frames"
    }

    if uploaded_file is not None:
        selected_option = st.selectbox("Select method", list(method_mapping.keys()))
        st.video(uploaded_file)

        method = method_mapping[selected_option]

        if st.button("Check"):
            with st.spinner("Checking..."):
                input_videofile_path = "uploaded_video.mp4"
                with open(input_videofile_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                fake_prob, real_prob, pred = predict_deepfake(input_videofile_path, method)

            if pred is None:
                st.error("Failed to detect DeepFakes")
            else:
                label = "real" if pred == 0 else "deepfaked"
                probability = real_prob if pred == 0 else fake_prob
                probability = round(probability * 100, 4)

                if pred == 0:
                    st.success(f"The video is {label}, with a probability of: {probability}%")
                    shutil.rmtree("./output")  
                else:
                    st.error(f"The video is {label}, with a probability of: {probability}%")
                    shutil.rmtree("./output")  

if __name__ == "__main__":
    main()
