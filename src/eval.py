from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, precision_recall_curve
import numpy as np
import logging



def best_threshold(model, X_val, y_val):

    y_probs = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    return thresholds[np.argmax(f1_scores)]
  


def evaluate(model, x_test, y_test, threshold, logger=None):
    y_probs = model.predict_proba(x_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    f1_test = f1_score(y_test, y_pred)
    
    logger.info(f"best threshold: {threshold:.2f}")
    logger.info(f"Final F1: {f1_test:.2f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_probs)}")
    logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))
    



