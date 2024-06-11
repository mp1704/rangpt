pipeline {
    agent any
    
    options{
        // Max number of build logs to keep and days to keep
        buildDiscarder(logRotator(numToKeepStr: '5', daysToKeepStr: '5'))
        // Enable timestamp at each job in the pipeline
        timestamps()
    }

    environment{
        registry = 'mp1704/rangpt'
        registryCredential = 'dockerhub'      
    }

    stages {
        stage('Test') {
            agent {
                docker {
                    image 'python:3.10-slim' 
                }
            }
            steps {
                echo 'Simple QA API testing..'
                sh 'pip install --no-cache-dir --upgrade pip'
                sh 'pip install -r requirements.txt'
                sh 'python -m pytest'
            }
        }
        stage('Build and Push') {
            steps {
                script {
                    echo 'Building image for deployment..'
                    dockerImage = docker.build registry + ":$BUILD_NUMBER" 
                    echo 'Pushing image to dockerhub..'
                    docker.withRegistry( '', registryCredential ) {
                        // dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }
    }
}