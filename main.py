from keybert import KeyBERT
import pandas as pd
from sklearn.ensemble import IsolationForest
import spacy
import yake

class TransactionReportAnalyzer:
    def __init__(self):
        self.keyword_model = KeyBERT()
        self.nlp = spacy.load('en_core_web_sm')
        self.kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=2,
            dedupLim=0.3,
            top=20
        )

#Keyword extraction from reports
def extract_keywords(self, text):
    # Using multiple keyword extraction methods for robustness
    keybert_keywords = self.keyword_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        use_maxsum=True,
        nr_candidates=20
    )
    
    # Extract domain-specific keywords using YAKE
    yake_keywords = self.kw_extractor.extract_keywords(text)
    
    # Combine and normalize results
    combined_keywords = {
        'transaction_related': [],
        'error_indicators': [],
        'health_metrics': []
    }
    
    # Categorize keywords based on domain knowledge
    for keyword, score in keybert_keywords:
        if any(term in keyword.lower() for term in ['error', 'fail', 'invalid']):
            combined_keywords['error_indicators'].append((keyword, score))
        elif any(term in keyword.lower() for term in ['transaction', 'payment', 'transfer']):
            combined_keywords['transaction_related'].append((keyword, score))
        elif any(term in keyword.lower() for term in ['health', 'status', 'check']):
            combined_keywords['health_metrics'].append((keyword, score))
    
    return combined_keywords

#anamoly detection in projection reports
def detect_anomalies(self, transaction_data):
    # Configure anomaly detection model
    isolation_forest = IsolationForest(
        contamination=0.1,
        random_state=42
    )
    
    # Prepare numerical features for anomaly detection
    numerical_features = transaction_data.select_dtypes(
        include=['float64', 'int64']
    )
    
    # Fit and predict anomalies
    anomalies = isolation_forest.fit_predict(numerical_features)
    
    # Add anomaly flags to the original data
    transaction_data['is_anomaly'] = anomalies == -1
    
    return transaction_data

#Health check for reporting system
def analyze_health_checks(self, report_text, transaction_data):
    analysis_results = {
        'keywords': self.extract_keywords(report_text),
        'anomalies': self.detect_anomalies(transaction_data),
        'health_status': self.evaluate_health_status(transaction_data)
    }
    
    return analysis_results

def evaluate_health_status(self, data):
    health_metrics = {
        'total_transactions': len(data),
        'anomaly_count': sum(data['is_anomaly']),
        'error_rate': sum(data['is_anomaly']) / len(data),
        'status': 'HEALTHY'
    }
    
    # Define health status based on thresholds
    if health_metrics['error_rate'] > 0.1:
        health_metrics['status'] = 'CRITICAL'
    elif health_metrics['error_rate'] > 0.05:
        health_metrics['status'] = 'WARNING'
    
    return health_metrics
