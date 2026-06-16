import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ["fbirn", "hcp"]
models = ["mlp", "bolT"]

data = []
for dataset in datasets:
    for model in models:
        path = f"/data/users2/ppopov1/mlp-project/assets/logs/A_A_A_SwiFT-exp-{model}_defHP-{dataset}/runs.csv"
        df = pd.read_csv(path)
        df["Model"] = "meanMLP" if model == "mlp" else "BolT"
        df["Dataset"] = "FBIRN" if dataset == "fbirn" else "HCP"
        data.append(df)

swift_fbirn = [0.6886101961135864, 0.5319865345954895, 0.6640712022781372, 0.7272727489471436, 0.5785634517669678]
swift_fbirn_pretrained = [0.6468129754066467, 0.6430976390838623, 0.6557285785675049, 0.6756453514099121, 0.5578002333641052]
# swift_hcp = [0.9865319132804871]
swift_hcp = [0.9598715901374817, 0.9392510652542114, 0.9188316464424133, 0.9188640713691711, 0.9438581466674805]
data.append(pd.DataFrame({
    "test_score": swift_fbirn,
    "Model": "SwiFT",
    "Dataset": "FBIRN",
}))
data.append(pd.DataFrame({
    "test_score": swift_fbirn_pretrained,
    "Model": "SwiFT (pretrained)",
    "Dataset": "FBIRN",
}))
data.append(pd.DataFrame({
    "test_score": swift_hcp,
    "Model": "SwiFT",
    "Dataset": "HCP",
}))

data = pd.concat(data)

data.to_csv("swift_results.csv")

palette = { item: plt.cm.tab20(i) for i, item in enumerate(data["Model"].unique())}

sns.set_theme(
    style="whitegrid", 
    # font_scale = 1.5,
)

plt.figure(figsize=(4.5, 3))
# sns.swarmplot(x="Dataset", y="test_score", hue="Model", data=data, palette=palette)
sns.boxplot(x="Dataset", y="test_score", hue="Model", data=data, palette=palette, showfliers=True)
plt.title("Test scores")
plt.xlabel("Dataset")
plt.ylabel("ROC AUC")

# Display the plot
plt.tight_layout()  # This ensures the labels don't get cut off
plt.savefig(
    "swift_comparison_boxplot.svg",
    # "swift_comparison.png",
    # format="png",
    # dpi=300,
    bbox_inches='tight',
)