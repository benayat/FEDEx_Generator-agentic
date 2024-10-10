import pandas as pd
import pd_explain
import warnings
warnings.filterwarnings("ignore")
import asyncio
import time
from dotenv import load_dotenv
import os


async def main():
    # environment = os.getenv('ENVIRONMENT', 'development')  # Default to 'development'
    # if environment == 'development':
    #     load_dotenv('.env.development')
    # elif environment == 'production':
    #     load_dotenv('.env.production')
    # else:
    #     load_dotenv('.env')  # Fallback to a default .env file
    adults = pd.read_csv(r"../../Notebooks/Datasets/adult.csv")
    by_income = adults.groupby('workclass').mean()

    await by_income.explain(top_k=4)


if __name__ == '__main__':
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")