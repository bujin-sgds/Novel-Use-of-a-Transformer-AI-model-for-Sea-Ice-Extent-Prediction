# Example helper script (optional): convert results .pkl into a quick summary
import pickle, sys, os

def main(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    results = data.get('results', {})
    print("\n=== ZhiQiu Results Summary ===")
    for k in ['model_version','run_tag','best_val_loss','epochs_trained','rmse','mae','r2','sie_rmse','sie_mae']:
        if k in results:
            print(f"{k}: {results[k]}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/summary_from_pkl.py <path_to_plotting_data.pkl>")
        sys.exit(1)
    main(sys.argv[1])