-- schema.sql
-- Table creation script for Sentiment Analysis Web Application

-- Create database if it does not exist
CREATE DATABASE IF NOT EXISTS sentiment_db;
USE sentiment_db;

-- Table to store sentiment predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    review_text TEXT NOT NULL,
    predicted_sentiment VARCHAR(15) NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

