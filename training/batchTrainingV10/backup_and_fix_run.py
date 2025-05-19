import wandb
import pandas as pd

# Skript ke korekci chybně vypočítaných metrik korelace u modelů
# wandering-yogurt-282, different-silence-283 a dainty-voice-284
# Docházelo k dělení metriky /len(val_loader) místo /len(trai_loader) při batch=16,
# tudíž byly všechny metriky 8x větší -> proto se zde dělí osmi.
# Dále pak provedený kontrolní run se správným kódem - sedí

# 🧾 PARAMETRY
entity = "bkubes_masters_thesis"
source_project = "konvid-test-2"
backup_project = "konvid-test-2-backup"
run_id = "sx2exops"  # ID běhu, který chceš opravit

# ✏️ Metriky, které chceme opravit dělením osmi
metrics_to_fix = ["avg_train_pearson_corr", "avg_train_r2_score", "avg_train_spearman_corr"]

# Přístup k API
api = wandb.Api()
source_run = api.run(f"{entity}/{source_project}/{run_id}")
history_df = source_run.history()
config = source_run.config
summary = dict(source_run.summary)
original_name = source_run.name
tags = list(source_run.tags)
group = source_run.group

def backup():
    wandb.init(project=backup_project, entity=entity, config=config, group=group)
    wandb.run.name = f"{original_name}_backup"
    wandb.run.tags = tags  # ← přenesení tagů
    wandb.run.save()

    for i, row in history_df.iterrows():
        step = int(row["_step"]) if "_step" in row else i
        log_data = {
            k: v for k, v in row.items()
            if isinstance(v, (int, float)) and not pd.isna(v)
        }
        wandb.log(log_data, step=step)

    for key, value in summary.items():
        if isinstance(value, (int, float, str)):
            wandb.run.summary[key] = value

    wandb.finish()
    print(f"✅ Běh {run_id} zálohován do projektu '{backup_project}' jako '{original_name}_backup'.")

def fix():
    wandb.init(project=source_project, entity=entity, config=config, group=group)
    wandb.run.name = f"{original_name}_fixed"
    wandb.run.tags = tags  # ← přenesení tagů
    wandb.run.save()

    for metric in metrics_to_fix:
        wandb.define_metric(metric, step_metric="_step", overwrite=True)

    for i, row in history_df.iterrows():
        if "_step" not in row:
            continue

        step = int(row["_step"])
        log_data = {"_step": step}

        for k, v in row.items():
            if isinstance(v, (int, float)) and not pd.isna(v):
                if k in metrics_to_fix:
                    log_data[k] = v / 8.0
                else:
                    log_data[k] = v

        wandb.log(log_data)

    for k, v in summary.items():
        if isinstance(v, (int, float)):
            wandb.run.summary[k] = v / 8.0 if k in metrics_to_fix else v
        elif isinstance(v, str):
            wandb.run.summary[k] = v

    wandb.finish()
    print(f"✅ Opravený běh '{original_name}_fixed' byl vytvořen v projektu '{source_project}'")

backup()
fix()
