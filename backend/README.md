## project structure

``
/ai-design-app
│── /backend                # Backend (Flask API, AI processing)
│   ├── /models             # AI models & scripts
│   ├── /routes             # API endpoints
│   ├── /services           # External API calls (OpenAI, AutoCAD, etc.)
│   ├── /database           # Database models & connections
│   ├── config.py           # App configuration settings
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   ├── .env                # Environment variables
│
│── /frontend               # Frontend (React.js)
│   ├── /src
│   │   ├── /components     # React components (UI elements)
│   │   ├── /pages          # Page components (Home, Dashboard)
│   │   ├── /services       # API call functions
│   │   ├── App.js          # Main React component
│   │   ├── index.js        # React entry point
│   ├── public              # Static assets
│   ├── package.json        # Frontend dependencies
│   ├── .env                # Frontend environment variables
│
│── /docker                 # Docker setup (optional)
│   ├── Dockerfile.backend  # Dockerfile for Flask API
│   ├── Dockerfile.frontend # Dockerfile for React frontend
│   ├── docker-compose.yml  # Compose file for running both services
│
│── /deployment             # Deployment scripts
│   ├── deploy_aws.sh       # AWS deployment script
│   ├── deploy_gcloud.sh    # Google Cloud deployment script
│
│── README.md               # Project documentation
│── .gitignore              # Ignore unnecessary files
``