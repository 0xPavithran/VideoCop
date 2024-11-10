**VideoCop** tackles the growing concern of **deepfake** content in the digital age. With the rise of synthetic media, it has become increasingly difficult to differentiate between genuine and manipulated videos. This poses significant challenges across various sectors

### Key Features:
1. **Deepfake Detection**: The app analyzes uploaded video files and uses state-of-the-art deepfake detection models to classify each frame as either "real" or "fake." This is done by examining facial features and patterns in the video frames.
   
2. **Face Detection**: Using **MTCNN** (Multi-task Cascaded Convolutional Networks), VideoCop detects and extracts faces from video frames. This helps focus the deepfake analysis on facial regions, improving the accuracy and reliability of the detection process.

3. **Real-Time Results**: The app processes videos and provides immediate feedback on whether the video is genuine or manipulated. Users are presented with probabilities to assess the likelihood of a video being a deepfake.

4. **User-Friendly Interface**: Built with **Streamlit**, VideoCop provides an easy-to-use web interface that allows users to upload videos and select detection methods with just a few clicks.

5. **Efficient Video Analysis**: The app works on popular video formats (e.g., MP4) and provides insights into the likelihood of deepfake content based on various visual cues. Users can easily understand the results through a clear display of probabilities and a labeled output (either "real" or "fake").
   
7. **Test the  VideoCop** : https://videocop.streamlit.app/ 

### How It Works:
1. **Upload a Video**: Users upload a video file via the simple file uploader interface.
2. **Face Extraction**: The system extracts faces from the video using MTCNN to focus on key regions.
3. **Deepfake Detection**: The extracted faces are passed through a pre-trained deepfake detection model to classify each frame as real or fake.
4. **Display Results**: After processing, the app displays the deepfake probability, showing whether the video is authentic or manipulated.

### Technologies Used:
- **Streamlit**: For building the interactive web interface.
- **MTCNN**: For face detection and extraction.
- **DeepFake Detection Models**: For analyzing video frames and detecting manipulated content.
- **Torch and TorchVision**: For running deep learning models for deepfake detection.
- **Shutil**: For handling file uploads and temporary file management.

