import os
import glob
import pandas as pd

SEP = ";"

# Кого хотим в core (логические имена)
CORE_LOGICAL = ["EURUSD", "USDKZT", "DXY", "BRENT", "VIX"]  # можешь потом убрать SP500 или VIX

# Синонимы на случай разных названий столбцов
ALIASES = {
    "EURUSD": ["EURUSD", "EURUSD=X"],
    "USDKZT": ["USDKZT", "USDKZT=X"],
    "DXY":    ["DXY", "DX-Y.NYB"],
    "BRENT":  ["BRENT", "BZ=F", "Brent"],
    "VIX":    ["VIX", "^VIX"],
}

def find_csv_candidates():
    # берём все csv в текущей папке, кроме новостного
    files = [f for f in glob.glob("*.csv") if "news" not in f.lower()]
    return files

def load_any_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=SEP)

    # приводим дату: либо есть колонка date, либо первая колонка — дата
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
    else:
        first = df.columns[0]
        dt = pd.to_datetime(df[first], errors="coerce")
        if dt.notna().mean() > 0.9:
            df[first] = dt
            df = df.dropna(subset=[first]).sort_values(first).set_index(first)
            df.index.name = "date"
        else:
            # fallback: пробуем как index_col=0
            df2 = pd.read_csv(path, sep=SEP, index_col=0)
            df2.index = pd.to_datetime(df2.index, errors="coerce")
            df2 = df2.dropna()
            df2.index.name = "date"
            df = df2.sort_index()

    return df

def pick_col(df: pd.DataFrame, logical: str):
    for name in ALIASES.get(logical, [logical]):
        if name in df.columns:
            return name
    return None

def main():
    csvs = find_csv_candidates()
    if not csvs:
        raise FileNotFoundError("В папке нет CSV файлов (кроме news). Сохрани свои датасеты в эту папку.")

    # Загружаем все кандидаты
    loaded = []
    for f in csvs:
        try:
            d = load_any_csv(f)
            loaded.append((f, d))
        except Exception as e:
            print(f"Skip {f}: {e}")

    if not loaded:
        raise RuntimeError("Не удалось прочитать ни один CSV. Проверь разделитель ';' и формат дат.")

    # Ищем, где лежат EURUSD/USDKZT и где лежат market indicators
    fx_df = None
    market_df = None

    for fname, df in loaded:
        has_eurusd = pick_col(df, "EURUSD") is not None
        has_usdkzt = pick_col(df, "USDKZT") is not None
        has_dxy = pick_col(df, "DXY") is not None

        # FX файл: есть EURUSD и USDKZT
        if fx_df is None and has_eurusd and has_usdkzt:
            fx_df = df
            fx_name = fname

        # Market файл: есть DXY и BRENT (или VIX/SP500)
        if market_df is None and has_dxy:
            market_df = df
            market_name = fname

    if fx_df is None:
        raise RuntimeError("Не найден файл с EURUSD и USDKZT. Убедись, что fx_multimodal_dataset_2010_present.csv лежит в этой папке.")
    if market_df is None:
        raise RuntimeError("Не найден файл с DXY (market indicators). Убедись, что market dataset сохранён в CSV в этой папке.")

    print("FX source:", fx_name)
    print("Market source:", market_name)

    # Собираем core из FX
    fx_cols = {}
    for logical in ["EURUSD", "USDKZT"]:
        real = pick_col(fx_df, logical)
        if real is None:
            raise RuntimeError(f"В FX датасете нет колонки {logical}. Доступные колонки: {list(fx_df.columns)[:20]} ...")
        fx_cols[logical] = real

    core_fx = fx_df[list(fx_cols.values())].rename(columns={v: k for k, v in fx_cols.items()})

    # Собираем core из Market
    market_needed = ["DXY", "BRENT", "VIX"]
    market_cols = {}
    for logical in market_needed:
        real = pick_col(market_df, logical)
        if real is not None:
            market_cols[logical] = real

    if "DXY" not in market_cols or "BRENT" not in market_cols:
        raise RuntimeError(
            "В market датасете должны быть хотя бы DXY и BRENT. "
            f"Найдено: {market_cols}. Проверь названия колонок в market CSV."
        )

    core_mkt = market_df[list(market_cols.values())].rename(columns={v: k for k, v in market_cols.items()})

    # Объединяем по дате (inner = только пересечение дат)
    merged = core_fx.join(core_mkt, how="inner")

    # Если хочешь сохранить ровно 5–6 колонок, оставим только CORE_LOGICAL которые реально есть
    final_cols = [c for c in CORE_LOGICAL if c in merged.columns]
    final = merged[final_cols].copy()

    # Статистика до/после
    before_fx = fx_df.shape[1]
    before_mkt = market_df.shape[1]
    after = final.shape[1]

    final = final.sort_index()
    final.to_csv("fx_core_dataset.csv", sep=SEP)
    final.to_excel("fx_core_dataset.xlsx")

    print("\n=== DONE ===")
    print(f"FX features before: {before_fx}")
    print(f"Market features before: {before_mkt}")
    print(f"Core features after preprocessing: {after}")
    print("Final columns:", final_cols)
    print("Rows:", final.shape[0])
    print(final.head())

if __name__ == "__main__":
    main()
