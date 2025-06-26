import numpy as np
import json
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Embedding Storage & Loading
# -----------------------------
def save_embeddings(embeddings_dict: Dict[str, List[np.ndarray]], filename: str):
    """
    Save embeddings to a JSON file. Converts numpy arrays to lists for serialization.
    """
    serializable = {k: [emb.tolist() for emb in v] for k, v in embeddings_dict.items()}
    with open(filename, 'w') as f:
        json.dump(serializable, f)

def load_embeddings(filename: str) -> Dict[str, List[np.ndarray]]:
    """
    Load embeddings from a JSON file. Converts lists back to numpy arrays.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return {k: [np.array(emb) for emb in v] for k, v in data.items()}

# -----------------------------
# Distance Functions
# -----------------------------
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# -----------------------------
# Best Match Logic
# -----------------------------
def get_best_match(
    input_embedding: np.ndarray,
    known_embeddings_dict: Dict[str, List[np.ndarray]],
    threshold: float = 0.6,
    distance_metric: str = 'euclidean',
    use_mean_vector: bool = False
) -> Tuple[Optional[str], float, float]:
    """
    Find the best match for the input embedding among all known embeddings.
    Returns (best_name, best_distance, confidence_score).
    """
    if distance_metric == 'euclidean':
        dist_func = euclidean_distance
    elif distance_metric == 'cosine':
        dist_func = cosine_distance
    else:
        raise ValueError("distance_metric must be 'euclidean' or 'cosine'")

    best_name = None
    best_distance = float('inf')
    confidence = 0.0

    for name, embeddings in known_embeddings_dict.items():
        if use_mean_vector:
            mean_emb = np.mean(embeddings, axis=0)
            d = dist_func(input_embedding, mean_emb)
            distances = [d]
        else:
            distances = [dist_func(input_embedding, emb) for emb in embeddings]
        min_dist = min(distances)
        if min_dist < best_distance:
            best_distance = min_dist
            best_name = name
            confidence = 1 - min_dist  # For logging/visualization (not a probability)

    if best_distance < threshold:
        return best_name, best_distance, confidence
    else:
        return None, best_distance, confidence

# -----------------------------
# Example: Registration Helper
# -----------------------------
def register_person(name: str, image_list: List, embedding_model, embeddings_dict: Dict[str, List[np.ndarray]]):
    """
    Compute and store multiple embeddings for a person.
    image_list: list of images (numpy arrays or file paths)
    embedding_model: function that returns embedding for an image
    embeddings_dict: dict to update
    """
    embeddings = []
    for img in image_list:
        emb = embedding_model(img)
        embeddings.append(emb)
    embeddings_dict[name] = embeddings
    return embeddings_dict

# -----------------------------
# (Optional) FAISS/Annoy Integration Placeholder
# -----------------------------
# For large datasets, you can add FAISS/Annoy index building and search here.
# See the README for details. 