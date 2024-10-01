import pandas as pd
import pd_explain
import warnings
warnings.filterwarnings("ignore")
import asyncio
import time

async def main():
    adults = pd.read_csv(r"../../Notebooks/Datasets/adult.csv")
    low_income = adults[adults['label'] == '<=50K']
    await low_income.explain(top_k=4)

if __name__ == '__main__':
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")