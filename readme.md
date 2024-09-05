# Flask Machine Learning Prediction App

This Flask application loads a machine learning model and provides prediction results for new data.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/JayathriRanasinghe/road-project-backend.git
    cd road-project-backend
    ```

2. **Run the setup script**:
    ```bash
    pip install -r requirements.txt
    ```

    ```bash
    python app.py
    ```

This will create a virtual environment, install the dependencies, and run the Flask app.

## API Usage

- **URL**: `http://127.0.0.1:5000/predict`
- **Method**: POST
- **Request Body**:
    ```json
    {
        "Time": 120,
        "sta": 1,
        "wether": 0,
        "week_or_weekend": 0
    }
    ```
- **Response**:
    ```json
    {
        "prediction": 123
    }
    ```
