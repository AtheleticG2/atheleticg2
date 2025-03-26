# Atheletics G2
# Athlete Assist - Pose Estimation & Analysis

## Overview
Athlete Assist is an AI-powered sports analysis system that uses advanced pose estimation to provide automated scoring and feedback for athletes. The system supports multiple sports and offers real-time performance analysis based on predefined biomechanical criteria. This project features a cloud-deployed Streamlit web interface for easy access and usage.

## 🏆 Supported Sports
- Sprint Start
- Sprint Running
- Long Jump
- High Jump
- Shot Put
- Discus Throw
- Javelin Throw

## ✨ Key Features
- **Advanced Pose Estimation**: Utilizes YOLOv11-Pose for accurate body tracking
- **Automated Scoring**: Evaluates performance based on sport-specific criteria
- **Real-time Analysis**: Provides immediate feedback on athlete technique
- **Multi-sport Support**: Comprehensive analysis across 7 different athletic events
- **Cloud-hosted Web Interface**: Access the system from anywhere via Hugging Face, Google Cloud Run or a locally run Docker container
- **SCRUM Workflow**: Developed using agile methodologies with the creator acting as Product Owner

## 🛠️ Technologies Used
- YOLOv11-Pose for skeletal tracking and pose estimation
- Python for backend development and analysis algorithms
- Streamlit for web interface
- Docker for containerization
- Azure, Hugging Face, and Google Cloud Run for cloud deployment
- Git for version control and collaboration

## 🖥️ Installation & Setup

### Local Development
```bash
# Clone the repository
git clone https://github.com/AtheleticG2/atheleticg2.git
cd atheleticg2

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app locally
streamlit run app.py
```

### Cloud Deployment Options

#### 1. Hugging Face Deployment
1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Create a new Space:
   - Select "Streamlit" as the SDK
   - Choose appropriate visibility settings
3. Upload the project files or connect to the GitHub repository
4. Your app will be available at `https://huggingface.co/spaces/yourusername/athlete-assist`

#### 2. Google Cloud Run Deployment
1. Create a Google Cloud account and enable necessary APIs:
   - Cloud Run API
   - Container Registry API
2. Deploy using the pre-built Docker image:
   ```
   Image: sathv0/athleticsg2:latest
   Port: 8501 (default Streamlit port)
   ```
3. Configure settings:
   - Allow unauthenticated invocations (for public access)
   - Set appropriate memory and CPU allocation
4. Deploy and access via the provided URL (e.g., `https://athlete-assist-xyz.a.run.app`)

## 🚀 Usage
1. Access the web interface (local or cloud-hosted)
2. Upload a video of an athlete performing one of the supported sports
3. Select the sport type from the dropdown menu
4. Click "Analyze" to process the video
5. View the automated scoring and technique analysis
6. Download the results report (optional)

## 📊 Performance Metrics
- Pose estimation accuracy: 90%+ on provided datasets
- Processing speed: 20-30 FPS for real-time analysis
- Cloud response time: ~5-8 minutes for video upload and processing


## 👥 Team & Roles
- Developed as a team project with SCRUM methodology
- Yaya Zhang, Joshua Ogunleye, Sathvika Alla

## 🔗 Links
- Live Demo: [Hugging Face Deployment](https://huggingface.co/spaces/yourusername/athlete-assist)
- GitHub Repository: [AtheleticG2/atheleticg2](https://github.com/AtheleticG2/atheleticg2)
- Docker Image: `sathv0/athleticsg2:latest`

## 🔄 Future Improvements
- [ ] Add support for additional sports
- [ ] Implement user accounts for saving and tracking progress
- [ ] Enhance visualization of analysis
- [ ] Develop mobile application for on-field use
- [ ] Integrate with wearable devices for additional data points

## 🙏 Acknowledgements
- YOLOv11-Pose developers for the pose estimation framework
- The sports science community for biomechanical insights
- Our advisors and mentors for guidance throughout the project

## Detailed Solution Documentation
[Athlete Assist functional analysis G2.pdf](https://github.com/user-attachments/files/19472854/Athlete.Assist.functional.analysis.G2.pdf)
