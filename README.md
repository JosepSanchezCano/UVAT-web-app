
# Web App

This repository provides a model for a web-based video annotation tool that leverages both SAM and VOS to deliver an AI-assisted annotation experience.

The app is built using Flask as a framework to manage frontend requests to the backend, which is written in Python. The backend handles video processing and AI model execution, while the frontend is more complex than usual, as it manages the display of masks and synchronization between the frontend and backend.

## Models

- **SAM2 (Segment Anything Model)** from Meta: Handles instance segmentation tasks.
- **Cutie (VOS Model)**: Propagates masks throughout the video.

## Features

- Upload videos (up to 5 minutes)
- Save annotations locally in COCO mask format
- Instance segmentation using point-based input
- Video object propagation (100 frames at a time)
- Frame-by-frame visualization
- Mask corrections
- Generate masks from scratch

## Upcoming Features

- Assigning classes to masks

## Installation

1. Create a virtual environment using your preferred tool (tested with Miniconda and Conda).
2. Install dependencies using pip:
   ```sh
   pip install -r requirements.txt
   ```
3. If Flask installation fails, follow the official Flask installation guide for your operating system.

4. To run the app you need the weights for each AI model. In SAM's case the model will download the weights upon execution of the model. This means that the first time that you execute the web app you will need to wait till the weights have been completly downloaded. The Cutie weights must be downloaded from their github at: https://github.com/hkchengrex/Cutie. Once downloaded please put the weights inside of the flaskr/weights/ folder.

## Running the Local Server

Since this app is not intended for deployment, it will be run locally using the Flask development server. To start the server:

1. Navigate to the folder containing the `flaskr` directory.
2. Run the following command:
   ```sh
   flask --app flaskr run --debug
   ```
3. Once the server starts, Ctrl+Click the provided URL to open the web app.

### Recommended Browser

Google Chrome is recommended, but any Chromium-based browser should work.

---
This project aims to provide an efficient and user-friendly AI-assisted annotation tool. Contributions and feedback are welcome!

 

