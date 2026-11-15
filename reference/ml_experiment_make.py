import os
import glob
import json
import time
import logging
import warnings
import numpy as np
import pandas as pd

# Sklearn Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Configuración de entorno
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IoTSecurityExperiment:
    def __init__(self, train_path, test_path, results_dir='./resultados_make'):
        self.train_path = train_path
        self.test_path = test_path
        self.results_dir = results_dir
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.target_columns = ['Class', 'Category', 'Attack']
        # Crear directorios de resultados
        os.makedirs(f"{self.results_dir}/figures", exist_ok=True)
        os.makedirs(f"{self.results_dir}/logs", exist_ok=True)

    def load_dataset(self, file_path, dataset_type="Train"):
        """
        Carga un archivo CSV único, maneja infinitos y realiza un shuffle completo.
        """
        logger.info(f"Loading {dataset_type} data from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise e
        
        # Reemplazar "–" (en-dash) y "-" (hyphen) por "Benign" en las columnas de 6 y 19 clases.
        cols_to_fix = ['Category', 'Attack']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = df[col].replace(['–', '-'], 'Benign')
                logger.info(f"Cleaned labels in column '{col}': replaced dashes with 'Benign'.")
        
        # Limpieza básica de infinitos y nulos
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        logger.info(f"Shuffling {dataset_type} dataset...")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"{dataset_type} dataset loaded and shuffled. Shape: {df.shape}")
        return df

    def get_configured_models(self):
        """
        Retorna el diccionario de modelos con los hiperparámetros exactos solicitados.
        """
        models = {
            # 1. Random Forest (RF)
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                criterion='gini',
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='sqrt',
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                warm_start=False,
                ccp_alpha=0.0,
                n_jobs=-1,  # Usar todos los núcleos
                random_state=42
            ),
            
            # 2. AdaBoost (AB) con base DecisionTree
            "AdaBoost": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=None), # Base estimator
                n_estimators=50,
                learning_rate=1.0,
                algorithm='SAMME',
                random_state=42
            ),
            
            # 3. Logistic Regression (LR)
            "LogisticRegression": LogisticRegression(
                penalty='l2',
                dual=False,
                tol=0.0001,
                C=1.0,
                fit_intercept=True,
                intercept_scaling=1,
                solver='lbfgs',
                max_iter=100,
                warm_start=False,
                n_jobs=-1,
                random_state=42
            ),
            
            # 4. Deep Neural Network (DNN) - Usando MLPClassifier (Sklearn)
            "DNN_MLP": MLPClassifier(
                hidden_layer_sizes=(32, 32, 32),
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=200,
                shuffle=True,
                tol=0.0001,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=10,
                random_state=42
            )
        }
        return models

    def run_complete_suite(self):
        # 1. Cargar Datos
        train_df = self.load_dataset(self.train_path, "Train")
        test_df = self.load_dataset(self.test_path, "Test")

        feature_cols = [col for col in train_df.columns if col not in self.target_columns]
        logger.info(f"Features detected: {len(feature_cols)}")
        logger.info(f"Target columns: {self.target_columns}")

        X_train = train_df[feature_cols]
        y_train_dict = {
            'binary': train_df['Class'],      # Benign / Attack
            '6class': train_df['Category'],   # DDoS, DoS, Recon...
            '19class': train_df['Attack']     # UDP_Flood, ...
        }
        
        X_test = test_df[feature_cols]
        y_test_dict = {
            'binary': test_df['Class'],
            '6class': test_df['Category'],
            '19class': test_df['Attack']
        }

        # Liberar memoria de dataframes originales
        del train_df, test_df

        # 3. Escalado (StandardScaler)
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Estructura para resultados finales
        full_results = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            },
            "results": {}
        }
        tasks = ['binary', '6class', '19class']

        for task in tasks:
            logger.info(f"\n{'='*40}\nSTARTING TASK: {task.upper()}\n{'='*40}")

            # Obtener vectores y
            y_train_raw = y_train_dict[task]
            y_test_raw = y_test_dict[task]
            
            # Label Encoding
            # Ajustamos (fit) SOLAMENTE con Train
            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_raw)

            # Transformamos Test. 
            # IMPORTANTE: Manejo de clases nuevas en Test (Data Leakage check)
            known_classes = set(le.classes_)
            test_unique = set(y_test_raw.unique())
            unknown_in_test = test_unique - known_classes
            
            if unknown_in_test:
                logger.warning(f"[{task}] Found unknown classes in Test set: {unknown_in_test}. Dropping these rows for evaluation.")
                # Crear máscara para quedarnos solo con lo conocido
                mask = y_test_raw.isin(known_classes)
                X_test_task_current = X_test_scaled[mask]
                y_test_task_current = y_test_raw[mask]
                # Ahora sí podemos transformar sin error
                y_test_encoded = le.transform(y_test_task_current)
            else:
                X_test_task_current = X_test_scaled
                y_test_encoded = le.transform(y_test_raw)

            class_names = le.classes_.tolist()
            logger.info(f"[{task}] Classes ({len(class_names)}): {class_names}")

            # Inicializar resultados para esta tarea
            full_results["results"][task] = {}
            models = self.get_configured_models()

            # 4. Ciclo de Entrenamiento y Evaluación
            for model_name, model in models.items():
                logger.info(f"Training {model_name}...")
                start_time = time.time()
            
                # Entrenar
                model.fit(X_train_scaled, y_train_encoded)
                train_time = time.time() - start_time
            
                # Predecir
                logger.info(f"Evaluating {model_name} on Held-Out Test Set...")
                y_pred = model.predict(X_test_task_current)
            
                # Métricas Standard (Weighted)
                acc = accuracy_score(y_test_encoded, y_pred)
                f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')
                prec_weighted = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                rec_weighted = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
                
                # Métricas MACRO (Para ver desempeño real en clases pequeñas)
                f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
                prec_macro = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
                rec_macro = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)

                mcc = matthews_corrcoef(y_test_encoded, y_pred)
                cm = confusion_matrix(y_test_encoded, y_pred)
                
                full_results["results"][task][model_name] = {
                    "metrics": {
                        "accuracy": acc,
                        "f1_weighted": f1_weighted,
                        "f1_macro": f1_macro,
                        "prec_weighted": prec_weighted,
                        "prec_macro": prec_macro,
                        "recall_weighted": rec_weighted,
                        "recall_macro": rec_macro,
                        "mcc": mcc,
                        "train_time_sec": train_time
                    },
                    "confusion_matrix": cm.tolist(),
                    "classification_report": classification_report(y_test_encoded, y_pred, target_names=class_names, output_dict=True)
                }
                logger.info(f"[{task}] Final {model_name} - Acc: {acc:.4f} | Weighted F1: {f1_weighted:.4f} | Macro F1: {f1_macro:.4f} | Weighted recall: {rec_weighted:.4f} | Macro recall: {rec_macro:.4f} | Weighted precision: {prec_weighted:.4f} | Macro precision: {prec_macro:.4f}" )
        
        # 5. Guardar JSON final
        json_path = f"{self.results_dir}/multitask_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"\nAll tasks completed. Results saved to {json_path}")

def find_csv_in_dir(directory):
    """Busca el primer archivo CSV en un directorio."""
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No .csv files found in {directory}")
    if len(files) > 1:
        logger.info(f"Multiple CSVs found in {directory}. Using: {files[0]}")
    return files[0]
        
if __name__ == "__main__":
    base_dir = os.getcwd() # Directorio actual
    
    train_dir_path = os.path.join(base_dir, 'train')
    test_dir_path = os.path.join(base_dir, 'test')

    TRAIN_FILE = find_csv_in_dir(train_dir_path)
    TEST_FILE = find_csv_in_dir(test_dir_path)  
    logger.info(f"Train File selected: {TRAIN_FILE}")
    logger.info(f"Test File selected: {TEST_FILE}")
    
    experiment = IoTSecurityExperiment(TRAIN_FILE, TEST_FILE)
    experiment.run_complete_suite()
    