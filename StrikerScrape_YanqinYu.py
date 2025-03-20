import pandas as pd
import os
import requests
from bs4 import BeautifulSoup


def scrape_high_scoring_seasons(player_name, player_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(player_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    data_div = soup.find("div", id="div_stats_shooting_dom_lg")
    if not data_div:
        print("Stats not found")
        return

    table_html = str(data_div.find("table"))
    df_list = pd.read_html(table_html, header=1)
    if not df_list:
        print("Not found")
        return

    df = df_list[0]

    df = df[df["Season"].notna()]

    df = df[df["Season"].str.match(r"^\d{4}-\d{4}$")]

    df["Gls"] = pd.to_numeric(df["Gls"], errors="coerce")

    # Extract season start year, need to be later than 2017
    df["Start_Year"] = df["Season"].str[:4].astype(int)

    # filter
    df_filtered = df[
        (df["Country"] == "eng ENG")
        & (df["Comp"] == "1. Premier League")
        & (df["Gls"] > 15)
        & (df["Season"] != "2024-2025")
        & (df["Start_Year"] >= 2017)
    ]

    if df_filtered.empty:
        print("Did not find stats that meet the filter reqs")
        return

    df_filtered = df_filtered.copy()
    df_filtered["Player"] = player_name

    df_filtered = df_filtered.drop(columns=["Start_Year"])

    output_path = f"Striker_csv/Testing_Set/{player_name}_high_scoring_seasons.csv"
    df_filtered.to_csv(output_path, index=False)
    print(f"CSV saved：{output_path}")
    print(df_filtered)


if __name__ == "__main__":
    player_name = "lukaku"
    player_url = "https://fbref.com/en/players/5eae500a/Romelu-Lukaku"
    scrape_high_scoring_seasons(player_name, player_url)
