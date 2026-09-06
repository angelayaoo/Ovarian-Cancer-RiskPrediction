"""Regenerate data/processed/* from data/raw/*.

Produces, byte-for-byte, the harmonized feature/label files used by every
analysis in the BIBM paper:

  chinese_train_cleaned.csv   from chinese training dataset(All Raw Data).csv
  west_china_processed.csv    from west_china.xlsx (Cancer + Cysts sheets)
  japan_processed.csv         from japanese.xlsx (Healthy women + Patients)

Run from the repository root:
  python src/clean_data.py
"""
import os
import pandas as pd

RAW = 'data/raw'
PROC = 'data/processed'


def main():
    os.makedirs(PROC, exist_ok=True)

    # ---------- training (Changzhou) ----------
    train = pd.read_csv(
        os.path.join(RAW, 'chinese training dataset(All Raw Data).csv'),
        encoding='latin-1')
    train.columns = [str(c).strip().lower() for c in train.columns]
    if 'type' in train.columns:
        train = train.rename(columns={'type': 'label'})
    # raw coding: 1 = benign (178), -1/0 = ovarian cancer (171)
    train['label'] = train['label'].map({1: 1, -1: 0, 0: 0}).astype(int)
    train.to_csv(os.path.join(PROC, 'chinese_train_cleaned.csv'), index=False)

    # ---------- West China (188 cancer + 192 cysts) ----------
    xls = pd.ExcelFile(os.path.join(RAW, 'west_china.xlsx'))
    cancer = xls.parse('Ovarian Cancer')
    cysts = xls.parse('Ovarian Cysts')
    rename = {
        'Preoperative serum CA125 (U/ML)': 'ca125',
        'Preoperative serum HE4 (pmol/L)': 'he4',
        'Menopausal status': 'menopause',
    }
    cancer = cancer.rename(columns=rename)
    cysts = cysts.rename(columns=rename)
    cancer['label'] = 1
    cysts['label'] = 0
    west = pd.concat([cancer, cysts], ignore_index=True)
    west = west[['age', 'ca125', 'he4', 'menopause', 'label']]
    west.to_csv(os.path.join(PROC, 'west_china_processed.csv'), index=False)

    # ---------- Japan (150 healthy + 27 patients) ----------
    xls = pd.ExcelFile(os.path.join(RAW, 'japanese.xlsx'))
    healthy = xls.parse('Healthy women')
    patients = xls.parse('Patients')
    healthy.columns = [str(c).strip().lower() for c in healthy.columns]
    patients.columns = [str(c).strip().lower() for c in patients.columns]
    healthy['label'] = 0
    patients['label'] = 1
    japan = pd.concat([healthy, patients], ignore_index=True, sort=False)
    japan.to_csv(os.path.join(PROC, 'japan_processed.csv'), index=False)

    print('processed files written:', os.listdir(PROC))


if __name__ == '__main__':
    main()
