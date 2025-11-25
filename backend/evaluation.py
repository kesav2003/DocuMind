import requests
import time
import csv
import os
from collections import defaultdict
from rouge import Rouge
from bert_score import score as bert_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
DATASET_DIR = "dataset"
CSV_PATH = "test_data.csv"
ACCURACY_THRESHOLD = 0.5  # ROUGE-1 F1 score threshold for an answer to be "correct"

def read_and_group_data():
    """Reads the CSV and groups questions by filename."""
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at '{CSV_PATH}'")
        return None
    
    data = defaultdict(list)
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) == 3:
                file_basename, question, ground_truth = row
                data[file_basename].append({
                    "question": question,
                    "reference_answer": ground_truth
                })
            else:
                print(f"Warning: Skipping malformed row {i+1} in {CSV_PATH}")

    print(f"Loaded {sum(len(v) for v in data.values())} questions for {len(data)} documents.")
    return data

def clear_knowledge_base():
    """Calls the /clear endpoint to reset the vector store."""
    print("\n--- Clearing Knowledge Base ---")
    try:
        response = requests.post(f"{API_BASE_URL}/clear")
        response.raise_for_status()
        print("Knowledge base cleared successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error clearing knowledge base: {e}")
        return False

def upload_document(file_path):
    """Uploads a single specified document."""
    print(f"--- Uploading Document: {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            files = {'files': (os.path.basename(file_path), f)}
            response = requests.post(f"{API_BASE_URL}/upload", files=files)
            response.raise_for_status()
            print("Document uploaded and indexed successfully.")
            time.sleep(2)
            return True
    except FileNotFoundError:
        print(f"ERROR: Document not found at '{file_path}'")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error uploading document: {e}")
        return False

def run_evaluation(grouped_data):
    """Runs the full evaluation pipeline on the grouped dataset."""
    print("\n--- Starting Full Evaluation ---")
    
    all_rouge_scores = []
    all_bert_scores = []
    all_bleu_scores = []
    correct_predictions = 0
    
    rouge = Rouge()
    smoother = SmoothingFunction().method1 # For BLEU score

    total_question_count = sum(len(v) for v in grouped_data.values())
    processed_question_count = 0
    dataset_files = os.listdir(DATASET_DIR)

    for doc_basename, qa_pairs in grouped_data.items():
        print(f"\n{'='*50}")
        print(f"Processing Document Base Name: {doc_basename}")
        print(f"{ '='*50}")

        full_filename = next((f for f in dataset_files if os.path.splitext(f)[0] == doc_basename), None)
        if not full_filename:
            print(f"  ERROR: Document for base name '{doc_basename}' not found. Skipping.")
            continue

        if not clear_knowledge_base() or not upload_document(os.path.join(DATASET_DIR, full_filename)):
            print(f"Skipping document {full_filename} due to setup error.")
            continue

        for qa_pair in qa_pairs:
            processed_question_count += 1
            question = qa_pair["question"]
            reference_answer = qa_pair["reference_answer"]
            
            print(f"\n({processed_question_count}/{total_question_count}) Evaluating Question: '{question[:80]}...'")

            try:
                response = requests.post(f"{API_BASE_URL}/query", json={"query": question}, stream=True)
                response.raise_for_status()
                candidate_answer = ""
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if '"token":' in line_str and '"sources":' not in line_str:
                            try:
                                import json
                                data = json.loads(line_str)
                                candidate_answer = data.get("token", "")
                            except json.JSONDecodeError:
                                continue
                
                if not candidate_answer.strip():
                    print("  SKIP: Generated answer is empty.")
                    continue

            except requests.exceptions.RequestException as e:
                print(f"  ERROR: API request failed: {e}")
                continue

            # --- Calculate all metrics ---
            try:
                # ROUGE-1
                rouge_1_scores = rouge.get_scores(candidate_answer, reference_answer)[0]['rouge-1']
                all_rouge_scores.append(rouge_1_scores)
                print(f"  - ROUGE-1: F1={rouge_1_scores['f']:.4f}, P={rouge_1_scores['p']:.4f}, R={rouge_1_scores['r']:.4f}")

                # Accuracy
                if rouge_1_scores['f'] > ACCURACY_THRESHOLD:
                    correct_predictions += 1

                # BERTScore
                P, R, F1 = bert_scorer([candidate_answer], [reference_answer], lang="en", verbose=False)
                bert_f1 = F1.mean().item()
                bert_p = P.mean().item()
                bert_r = R.mean().item()
                all_bert_scores.append({'f': bert_f1, 'p': bert_p, 'r': bert_r})
                print(f"  - BERTScore: F1={bert_f1:.4f}, P={bert_p:.4f}, R={bert_r:.4f}")

                # BLEU
                reference_tokens = [reference_answer.split()]
                candidate_tokens = candidate_answer.split()
                bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoother)
                all_bleu_scores.append(bleu_score)
                print(f"  - BLEU: {bleu_score:.4f}")

            except (ValueError, KeyError) as e:
                print(f"  SKIP: Could not calculate metric. Error: {e}")

    # --- Final Report ---
    print(f"\n{'='*60}")
    print("---           FINAL EVALUATION REPORT           ---")
    print(f"{ '='*60}")

    total_evaluated = len(all_rouge_scores)
    print(f"Successfully evaluated {total_evaluated} out of {total_question_count} questions.")

    if total_evaluated > 0:
        # ROUGE-1
        avg_rouge_f1 = sum(s['f'] for s in all_rouge_scores) / total_evaluated
        avg_rouge_p = sum(s['p'] for s in all_rouge_scores) / total_evaluated
        avg_rouge_r = sum(s['r'] for s in all_rouge_scores) / total_evaluated
        print(f"\nAverage ROUGE-1:")
        print(f"  - F1-Score:  {avg_rouge_f1:.4f}")
        print(f"  - Precision: {avg_rouge_p:.4f}")
        print(f"  - Recall:    {avg_rouge_r:.4f}")

        # Accuracy
        accuracy = (correct_predictions / total_evaluated) * 100
        print(f"\nAccuracy (ROUGE-1 F1 > {ACCURACY_THRESHOLD}):")
        print(f"  - {accuracy:.2f}% ({correct_predictions}/{total_evaluated} correct)")

        # BERTScore
        avg_bert_f1 = sum(s['f'] for s in all_bert_scores) / total_evaluated
        avg_bert_p = sum(s['p'] for s in all_bert_scores) / total_evaluated
        avg_bert_r = sum(s['r'] for s in all_bert_scores) / total_evaluated
        print(f"\nAverage BERTScore:")
        print(f"  - F1-Score:  {avg_bert_f1:.4f}")
        print(f"  - Precision: {avg_bert_p:.4f}")
        print(f"  - Recall:    {avg_bert_r:.4f}")

        # BLEU
        avg_bleu = sum(all_bleu_scores) / total_evaluated
        print(f"\nAverage BLEU Score:")
        print(f"  - {avg_bleu:.4f}")

if __name__ == "__main__":
    grouped_data = read_and_group_data()
    if grouped_data:
        run_evaluation(grouped_data)
