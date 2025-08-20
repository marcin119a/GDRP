import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_drug_availability(drugs_to_test, data_path="data/merged_df_all_drugs.parquet"):
    """
    Check which drugs from the list are available in the dataset.
    """
    print("=== Checking Drug Availability ===")

    # Load the dataset
    df = pd.read_parquet(data_path)

    available_drugs = []
    unavailable_drugs = []

    for drug in drugs_to_test:
        count = len(df[df['DRUG_NAME'] == drug])
        if count > 0:
            available_drugs.append((drug, count))
        else:
            unavailable_drugs.append(drug)

    print(f"\nAvailable drugs ({len(available_drugs)}):")
    print(f"{'Drug Name':<25} {'Count':<10}")
    print("-" * 35)
    for drug, count in sorted(available_drugs, key=lambda x: x[1], reverse=True):
        print(f"{drug:<25} {count:<10}")

    if unavailable_drugs:
        print(f"\nUnavailable drugs ({len(unavailable_drugs)}):")
        for drug in unavailable_drugs:
            print(f"  - {drug}")

    print(f"\nTotal samples for available drugs: {sum(count for _, count in available_drugs)}")

    return [drug for drug, _ in available_drugs]
